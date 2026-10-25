"""
Hybrid recommender (SVD + TF-IDF) with fake-interaction / bot-attack defense.

Dataset : MovieLens ml-latest-small  (links.csv, movies.csv, ratings.csv, tags.csv)
Pipeline: 1. detect suspicious (bot / shilling) users  -> remove their ratings + tags
          2. SVD matrix factorisation (collaborative)   -> trained on clean ratings only
          3. TF-IDF on genres + decade + tags (content) -> cold-start + low-support items
          4. blend: alpha = n_ratings / (n_ratings + k_blend)

Place at: cws/backend/recom_train/model.py
Usage (run from the folder that contains model.py):
    python model.py train                 # train + save hybrid_model.pkl
    python model.py eval                  # RMSE / MAE / Precision@10 / Recall@10
    python model.py attack                # simulate a bot attack, with and without defense
    python model.py recommend --user 5    # top-10 for an existing user
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split as sk_split
from sklearn.preprocessing import StandardScaler
from surprise import SVD, Dataset, Reader

SEED = 42
# model.py lives in cws/backend/recom_train/  ->  parents[2] == cws/
DATA_DIR = Path(__file__).resolve().parents[2] / "cwsdataset" / "ml-latest-small"
MODEL_PATH = Path(__file__).resolve().parent / "hybrid_model.pkl"
FAKE_BASE = 1_000_000  # ids >= this are simulated bots in the attack experiment


# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
def _read(path: Path, **kw) -> pd.DataFrame:
    """read_csv that survives a BOM / stray spaces in the header (common on Windows)."""
    df = pd.read_csv(path, encoding="utf-8-sig", **kw)
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df


def load_data(data_dir: Path = DATA_DIR):
    ratings = _read(data_dir / "ratings.csv")
    movies = _read(data_dir / "movies.csv")
    tags = _read(data_dir / "tags.csv")
    links = _read(data_dir / "links.csv", dtype=str)   # str keeps leading zeros of imdbId
    if "movieId" not in links.columns:
        print(f"WARNING: links.csv has columns {list(links.columns)} (no movieId) -> imdb/tmdb ids skipped")
        links = None
    else:
        links["movieId"] = pd.to_numeric(links.movieId, errors="coerce")
        links = links.dropna(subset=["movieId"]).astype({"movieId": int})
    ratings = ratings[ratings.movieId.isin(movies.movieId)].reset_index(drop=True)
    tags = tags[tags.movieId.isin(movies.movieId)].reset_index(drop=True)
    return ratings, movies, tags, links


# ----------------------------------------------------------------------------
# 2. BOT / FAKE-INTERACTION DETECTION
# ----------------------------------------------------------------------------
def build_user_features(ratings: pd.DataFrame, tags: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-user behavioural features commonly used in shilling-attack detection."""
    r = ratings.copy()
    item = r.groupby("movieId").rating.agg(item_mean="mean", item_cnt="count")
    r = r.join(item, on="movieId")

    r["dev"] = (r.rating - r.item_mean).abs() / r.item_cnt          # RDMA term
    r["extreme"] = r.rating.isin([r.rating.min(), r.rating.max()]).astype(float)
    r["niche"] = (r.item_cnt <= item.item_cnt.median()).astype(float)
    r["log_pop"] = np.log1p(r.item_cnt)
    r["hour"] = r.timestamp // 3600

    g = r.groupby("userId")
    n = g.size()
    feats = pd.DataFrame(
        {
            "log_n_ratings": np.log1p(n),
            "mean_rating": g.rating.mean(),
            "std_rating": g.rating.std(ddof=0),
            "pct_extreme": g.extreme.mean(),      # bots love 5.0 / 0.5
            "rdma": g.dev.sum() / n,              # Rating Deviation from Mean Agreement
            "pct_niche": g.niche.mean(),
            "mean_log_pop": g.log_pop.mean(),
            # share of all ratings that fall in the user's busiest hour (scripted bursts)
            "burst_frac": r.groupby(["userId", "hour"]).size().groupby("userId").max() / n,
        }
    )

    if tags is not None and len(tags):
        tg = tags.groupby("userId").tag.agg(
            n_tags="size", tag_unique=lambda s: s.str.lower().nunique() / len(s)
        )
        feats = feats.join(tg)
    if "n_tags" not in feats:
        feats["n_tags"], feats["tag_unique"] = 0, 1.0
    feats["n_tags"] = feats["n_tags"].fillna(0)
    feats["tag_unique"] = feats["tag_unique"].fillna(1.0)
    feats["log_n_tags"] = np.log1p(feats.pop("n_tags"))
    feats["n_ratings"] = n
    return feats


def detect_item_spikes(ratings: pd.DataFrame, window: int = 86400,
                       min_count: int = 10, same_frac: float = 0.8) -> set:
    """
    Target-item attack detector. A push/nuke attack shows up as many DIFFERENT users giving
    the SAME extreme rating (max or min) to one item inside a short window.
    Returns the set of userIds that took part in such a burst.
    """
    lo, hi = ratings.rating.min(), ratings.rating.max()
    suspects: set = set()
    for _, g in ratings.groupby("movieId"):
        if len(g) < min_count:
            continue
        g = g.sort_values("timestamp")
        ts = g.timestamp.to_numpy()
        end = np.searchsorted(ts, ts + window, side="right")
        cnt = end - np.arange(len(ts))
        i = int(cnt.argmax())
        if cnt[i] < min_count:
            continue
        win = g.iloc[i:end[i]]
        for val in (hi, lo):
            same = win[win.rating == val]
            if len(same) >= min_count and len(same) / len(win) >= same_frac:
                suspects |= set(same.userId)
    return suspects


def detect_bots(
    ratings: pd.DataFrame,
    tags: pd.DataFrame | None = None,
    z_thresh: float = 3.5,
    max_flag_frac: float = 0.10,
    min_ratings_for_rule: int = 10,
) -> pd.DataFrame:
    """
    Unsupervised bot detection.
      * Isolation Forest on behavioural features -> robust z-score (median/MAD)
      * item-burst detector (detect_item_spikes) -> catches push/nuke attacks even when
        the bots look 'normal' individually
      * hard rule: constant ratings over >= min_ratings_for_rule items
      * never flags more than `max_flag_frac` of users (safety cap)
    Returns the feature table with `anomaly_z` and boolean `is_bot`.
    """
    feats = build_user_features(ratings, tags)
    model_cols = [c for c in feats.columns if c != "n_ratings"]
    X = StandardScaler().fit_transform(feats[model_cols])

    iso = IsolationForest(n_estimators=300, random_state=SEED).fit(X)
    score = -iso.score_samples(X)                       # higher = more suspicious
    med = np.median(score)
    mad = np.median(np.abs(score - med)) or 1e-9
    z = 0.6745 * (score - med) / mad

    spike_users = detect_item_spikes(ratings)
    rule = ((feats.n_ratings >= min_ratings_for_rule) & (feats.std_rating == 0)).to_numpy().copy()
    rule |= np.asarray(feats.index.isin(spike_users))
    z = np.where(rule, z.max() + 1, z)                  # rule hits rank first
    flagged = (z > z_thresh) | rule

    cap = int(np.ceil(max_flag_frac * len(feats)))
    if flagged.sum() > cap:
        keep = np.argsort(-z)[:cap]
        flagged = np.zeros(len(feats), bool)
        flagged[keep] = True

    feats["anomaly_z"] = z
    feats["is_bot"] = flagged
    return feats.sort_values("anomaly_z", ascending=False)


# ----------------------------------------------------------------------------
# 3. HYBRID MODEL
# ----------------------------------------------------------------------------
def train_svd(ratings: pd.DataFrame, n_factors=100, n_epochs=30, lr=0.005, reg=0.05):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr, reg_all=reg, random_state=SEED)
    algo.fit(trainset)
    return algo, trainset


def build_documents(movies: pd.DataFrame, tags: pd.DataFrame) -> pd.Series:
    """One text 'document' per movie: genres (x2) + decade + user tags."""
    genres = (
        movies.genres.str.replace("(no genres listed)", "", regex=False)
        .str.lower().str.replace("-", "_", regex=False).str.replace("|", " ", regex=False)
    )
    year = pd.to_numeric(movies.title.str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
    decade = ("decade_" + (year // 10 * 10).astype("Int64").astype(str) + "s").where(year.notna(), "")

    t = tags.assign(tag=tags.tag.astype(str).str.lower().str.strip().str.replace(r"\s+", "_", regex=True))
    movie_tags = t.groupby("movieId").tag.apply(" ".join)
    tag_text = movies.movieId.map(movie_tags).fillna("")
    return (genres + " " + genres + " " + decade + " " + tag_text).str.strip()


class HybridRecommender:
    def __init__(self, k_blend=20, min_item_support=3, remove_bots=True, **svd_kwargs):
        self.k_blend = k_blend                    # bigger = trust content longer
        self.min_item_support = min_item_support  # items with fewer clean ratings -> content only
        self.remove_bots = remove_bots
        self.svd_kwargs = svd_kwargs

    # ---- training ----------------------------------------------------------
    def fit(self, ratings, movies, tags, links=None):
        self.bot_report = detect_bots(ratings, tags) if self.remove_bots else None
        self.bot_ids = set(self.bot_report.index[self.bot_report.is_bot]) if self.remove_bots else set()

        clean = ratings[~ratings.userId.isin(self.bot_ids)]
        clean_tags = tags[~tags.userId.isin(self.bot_ids)]
        self.clean_ratings = clean

        # collaborative part
        self.svd, self.trainset = train_svd(clean, **self.svd_kwargs)

        # content part
        self.movies = movies.reset_index(drop=True)
        self.movie_ids = self.movies.movieId.to_numpy()
        self.pos_of = {int(m): i for i, m in enumerate(self.movie_ids)}
        self.vectorizer = TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True)
        self.tfidf = self.vectorizer.fit_transform(build_documents(self.movies, clean_tags))

        # alignment between Surprise's inner item ids and movie_ids order
        self.inner_to_pos = np.array(
            [self.pos_of.get(int(self.trainset.to_raw_iid(i)), -1) for i in range(self.trainset.n_items)]
        )
        # item stats on clean data (support + Bayesian popularity)
        stats = clean.groupby("movieId").rating.agg(["sum", "count"]).reindex(self.movie_ids).fillna(0)
        self.item_cnt = stats["count"].to_numpy()
        m, C = clean.rating.mean(), 10
        self.pop_score = ((C * m + stats["sum"]) / (C + stats["count"])).to_numpy()

        self.user_hist = {u: dict(zip(g.movieId, g.rating)) for u, g in clean.groupby("userId")}
        self.links = links.set_index("movieId") if links is not None else None
        return self

    # ---- scoring helpers ---------------------------------------------------
    @staticmethod
    def _rank01(x):
        return pd.Series(x).rank(pct=True).to_numpy()

    def _svd_scores(self, user_id):
        """Vectorised SVD prediction for every movie (NaN = unknown / low-support item)."""
        iu = self.trainset.to_inner_uid(user_id)
        a = self.svd
        inner = self.trainset.global_mean + a.bu[iu] + a.bi + a.qi @ a.pu[iu]
        out = np.full(len(self.movie_ids), np.nan)
        ok = self.inner_to_pos >= 0
        out[self.inner_to_pos[ok]] = np.clip(inner[ok], 0.5, 5.0)
        out[self.item_cnt < self.min_item_support] = np.nan
        return out

    def _content_scores(self, hist):
        pos = [self.pos_of[m] for m in hist if m in self.pos_of]
        w = np.clip(np.array([hist[self.movie_ids[p]] for p in pos]) - 2.5, 0, None)
        if len(pos) == 0 or w.sum() == 0:
            return None
        profile = csr_matrix(w) @ self.tfidf[pos]          # weighted taste profile
        return linear_kernel(profile, self.tfidf).ravel()

    # ---- public API --------------------------------------------------------
    def recommend(self, user_id=None, n=10, liked: dict | None = None) -> pd.DataFrame:
        """
        user_id : existing user (optional)
        liked   : {movieId: rating} for a NEW user (cold-start) or extra fresh ratings
        """
        hist = dict(self.user_hist.get(user_id, {}))
        hist.update(liked or {})
        n_r = len(hist)

        content = self._content_scores(hist) if n_r else None
        svd = None
        try:
            svd = self._svd_scores(user_id) if user_id is not None else None
        except ValueError:                                  # user unknown to SVD
            svd = None

        if content is None and svd is None:                 # total cold start
            score = self._rank01(self.pop_score)
        else:
            c = self._rank01(content) if content is not None else self._rank01(self.pop_score)
            if svd is None:
                score = c
            else:
                s = self._rank01(np.where(np.isnan(svd), np.nanmin(svd), svd))
                alpha = n_r / (n_r + self.k_blend)
                score = np.where(np.isnan(svd), c, alpha * s + (1 - alpha) * c)

        score = np.array(score, dtype=float)
        seen_pos = [self.pos_of[m] for m in hist if m in self.pos_of]
        score[seen_pos] = -np.inf
        top = np.argsort(-score)[:n]

        out = self.movies.iloc[top][["movieId", "title", "genres"]].copy()
        out["score"] = score[top]
        if self.links is not None:
            out["imdbId"] = out.movieId.map(self.links.imdbId)
            out["tmdbId"] = out.movieId.map(self.links.tmdbId)
        return out.reset_index(drop=True)

    def similar_movies(self, movie_id: int, n=10) -> pd.DataFrame:
        p = self.pos_of[int(movie_id)]
        sims = linear_kernel(self.tfidf[p], self.tfidf).ravel()
        sims[p] = -1
        top = np.argsort(-sims)[:n]
        out = self.movies.iloc[top][["movieId", "title", "genres"]].copy()
        out["similarity"] = sims[top]
        return out.reset_index(drop=True)

    # ---- persistence -------------------------------------------------------
    def save(self, path: Path = MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "HybridRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)


# ----------------------------------------------------------------------------
# 4. EVALUATION
# ----------------------------------------------------------------------------
def evaluate_hybrid(ratings, movies, tags, links, k=10, n_users=150):
    train, test = sk_split(ratings, test_size=0.2, random_state=SEED)
    model = HybridRecommender().fit(train, movies, tags, links)

    known = test[test.userId.isin(model.user_hist) & test.movieId.isin(model.clean_ratings.movieId)]
    pred = np.array([model.svd.predict(u, i).est for u, i in zip(known.userId, known.movieId)])
    err = pred - known.rating.to_numpy()
    print(f"SVD  RMSE: {np.sqrt(np.mean(err**2)):.4f}   MAE: {np.mean(np.abs(err)):.4f}")

    rng = np.random.default_rng(SEED)
    users = rng.permutation([u for u in test.userId.unique() if u in model.user_hist])[:n_users]
    prec, rec = [], []
    for u in users:
        rel = set(test[(test.userId == u) & (test.rating >= 4)].movieId)
        if not rel:
            continue
        got = set(model.recommend(u, n=k).movieId)
        prec.append(len(rel & got) / k)
        rec.append(len(rel & got) / len(rel))
    print(f"Hybrid Precision@{k}: {np.mean(prec):.4f}   Recall@{k}: {np.mean(rec):.4f}")
    print(f"Users flagged as bots in training split: {len(model.bot_ids)}")


# ----------------------------------------------------------------------------
# 5. BOT-ATTACK SIMULATION (for testing the defense)
# ----------------------------------------------------------------------------
def inject_shilling_attack(ratings, target_movie, n_bots=50, filler_size=40,
                           attack="average", push=True, seed=SEED):
    """
    Adds fake users that all rate `target_movie` 5.0 (push) or 0.5 (nuke), plus filler items.
      random   : filler rated randomly
      average  : filler rated near each item's mean (harder to detect)
      bandwagon: popular items rated 5.0 so the bot looks like a normal fan
    Bots act in a short time window (scripted behaviour).
    """
    rng = np.random.default_rng(seed)
    stats = ratings.groupby("movieId").rating.agg(["mean", "count"])
    pool = stats.drop(index=target_movie, errors="ignore").sort_values("count", ascending=False).index[:300]
    t0 = int(ratings.timestamp.max()) + 1
    rows = []
    for b in range(n_bots):
        uid = FAKE_BASE + b
        filler = rng.choice(pool, size=filler_size, replace=False)
        if attack == "random":
            vals = rng.choice(np.arange(0.5, 5.01, 0.5), size=filler_size)
        elif attack == "bandwagon":
            vals = np.full(filler_size, 5.0)
        else:
            vals = np.clip(np.round((stats.loc[filler, "mean"].to_numpy() + rng.normal(0, 0.8, filler_size)) * 2) / 2, 0.5, 5.0)
        ts = t0 + rng.integers(0, 3600, size=filler_size + 1)
        rows += [(uid, int(m), float(v), int(t)) for m, v, t in zip(filler, vals, ts[:-1])]
        rows.append((uid, int(target_movie), 5.0 if push else 0.5, int(ts[-1])))
    fake = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    return pd.concat([ratings, fake], ignore_index=True)


def attack_experiment(ratings, movies, tags, links, n_bots=50, filler_size=40, attack="average"):
    rng = np.random.default_rng(SEED)
    stats = ratings.groupby("movieId").rating.agg(["mean", "count"])
    cand = stats[stats["count"].between(15, 80)].sort_values("mean")
    target = int(cand.index[0])                          # poorly rated, mid-popularity -> push target
    title = movies.set_index("movieId").title[target]
    rated = set(ratings[ratings.movieId == target].userId)
    sample = rng.choice([u for u in ratings.userId.unique() if u not in rated], size=100, replace=False)

    def avg_pred(df):
        algo, _ = train_svd(df)
        return float(np.mean([algo.predict(u, target).est for u in sample]))

    attacked = inject_shilling_attack(ratings, target, n_bots, filler_size, attack)
    report = detect_bots(attacked, tags)
    flagged = set(report.index[report.is_bot])
    fake = {u for u in report.index if u >= FAKE_BASE}
    tp, fp = len(flagged & fake), len(flagged - fake)

    base = avg_pred(ratings)
    no_def = avg_pred(attacked)
    with_def = avg_pred(attacked[~attacked.userId.isin(flagged)])

    print(f"\nTarget: {title}  (true mean {stats.loc[target, 'mean']:.2f}, {int(stats.loc[target, 'count'])} ratings)")
    print(f"Attack: {attack}, {n_bots} bots, {filler_size} filler items each\n")
    print("Avg predicted rating for target over 100 real users")
    print(f"  no attack                 : {base:.3f}")
    print(f"  attack, NO defense        : {no_def:.3f}   (shift {no_def - base:+.3f})")
    print(f"  attack, WITH bot filter   : {with_def:.3f}   (shift {with_def - base:+.3f})")
    print(f"\nDetection: caught {tp}/{len(fake)} bots, {fp} real users wrongly flagged")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "attack", "recommend"])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--user", type=int, default=1)
    ap.add_argument("--attack", choices=["random", "average", "bandwagon"], default="average")
    ap.add_argument("--bots", type=int, default=50)
    args = ap.parse_args()

    ratings, movies, tags, links = load_data(args.data_dir)
    print(f"{len(ratings):,} ratings | {ratings.userId.nunique()} users | {len(movies):,} movies | {len(tags):,} tags")

    if args.cmd == "train":
        model = HybridRecommender().fit(ratings, movies, tags, links)
        model.save()
        print(f"Flagged {len(model.bot_ids)} suspicious users:", sorted(model.bot_ids)[:20])
        print(model.bot_report.head(10).round(3).to_string())
        print(f"Saved -> {MODEL_PATH}")
    elif args.cmd == "eval":
        evaluate_hybrid(ratings, movies, tags, links)
    elif args.cmd == "attack":
        attack_experiment(ratings, movies, tags, links, n_bots=args.bots, attack=args.attack)
    else:
        model = HybridRecommender.load()
        print(model.recommend(args.user, n=10).to_string(index=False))