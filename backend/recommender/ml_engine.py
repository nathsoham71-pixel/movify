"""
Hybrid movie recommendation engine.

    Content engine        TF-IDF over movie genres + cosine similarity.
                          Works for any movie, even with zero ratings (cold start).
    Collaborative engine  Surprise SVD matrix factorisation, trained on the
                          sanitised ratings plus likes from trusted users.
    Taste profile         Built live from a user's likes/dislikes (no retraining):
                          a weighted sum of the liked movies' genre vectors.
    Hybrid                Candidates come from the content engine (movies similar
                          to the seed and to the user's taste), then each is
                          re-ranked by SVD prediction + seed similarity + taste.

The engine trains once on first use and is served from memory. Every
RECOMMENDER_RETRAIN_EVERY (settings, default 5) trusted like changes, a
background thread retrains it so the SVD model learns the new likes as
5-star / 1-star ratings.

Usage:
    from recommender.ml_engine import get_engine
    result = get_engine().recommend(user_id=1, movie_id=10)   # result.movies
    result = get_engine().for_you(user_id=1)
"""
import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from django.conf import settings
from django.db import connection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

from .bot_defense import get_training_dataframe, is_user_restricted
from .models import Like, Movie, Rating

logger = logging.getLogger(__name__)

NO_GENRES = "(no genres listed)"
RATING_SCALE = (0.5, 5.0)
DISLIKE_WEIGHT = -0.5


def retrain_every():
    """Trusted like changes between background retrains (0 disables retraining)."""
    return getattr(settings, "RECOMMENDER_RETRAIN_EVERY", 5)


class MovieNotFound(LookupError):
    pass


def _genre_tokens(genres):
    """'Sci-Fi|Film-Noir' -> ['sci-fi', 'film-noir'] (keeps multi-word genres intact)."""
    if not genres or genres == NO_GENRES:
        return []
    return [g.strip().lower() for g in genres.split("|") if g.strip()]


def _normalise_rating(values):
    low, high = RATING_SCALE
    return (np.asarray(values, dtype=float) - low) / (high - low)


class ContentEngine:
    """TF-IDF genre vectors. Similarities are computed per query, never as a full NxN matrix."""

    def __init__(self, movies_df):
        self.movie_ids = movies_df["movie_id"].to_numpy()
        self.row_of = {int(mid): row for row, mid in enumerate(self.movie_ids)}
        genres = movies_df["genres"].fillna("")
        self.genre_sets = [set(_genre_tokens(g)) for g in genres]
        self.genre_label = {}
        for value in genres:
            for name in value.split("|"):
                if name and name != NO_GENRES:
                    self.genre_label.setdefault(name.lower(), name)
        self.vectorizer = TfidfVectorizer(analyzer=_genre_tokens)
        self.matrix = self.vectorizer.fit_transform(genres)

    def similarities(self, movie_id):
        row = self.row_of.get(int(movie_id))
        if row is None:
            raise MovieNotFound(f"Movie {movie_id} does not exist.")
        # Rounded so movies with identical genre sets tie exactly.
        return np.round(cosine_similarity(self.matrix[row], self.matrix).ravel(), 6)


class CollaborativeEngine:
    """Surprise SVD trained on the bot-filtered ratings DataFrame."""

    def __init__(self, ratings_df, n_factors=100, n_epochs=20, random_state=42):
        reader = Reader(rating_scale=RATING_SCALE)
        data = Dataset.load_from_df(ratings_df[["user_id", "movie_id", "rating"]], reader)
        self.trainset = data.build_full_trainset()
        self.model = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=random_state)
        self.model.fit(self.trainset)

    def knows_user(self, user_id):
        try:
            self.trainset.to_inner_uid(int(user_id))
            return True
        except ValueError:
            return False

    def predict(self, user_id, movie_ids):
        return np.array([
            self.model.predict(int(user_id), int(mid), clip=True).est for mid in movie_ids
        ])


@dataclass
class TasteProfile:
    """A user's live genre preferences derived from their likes and dislikes."""

    sims: np.ndarray                 # cosine similarity of every movie to the taste vector
    n_likes: int
    n_dislikes: int
    genre_counts: Counter

    @property
    def n_signals(self):
        return self.n_likes + self.n_dislikes

    def weight(self, per_signal=0.08, cap=0.4):
        """How much of the hybrid score the taste term gets; grows with evidence."""
        return min(cap, per_signal * self.n_signals)


@dataclass
class Recommendations:
    movies: list
    strategy: str
    weights: dict
    taste: dict = field(default_factory=dict)


class HybridRecommender:
    def __init__(self, candidate_pool=50, taste_pool=30, for_you_pool=100,
                 cf_weight=0.7, popularity_prior=20, defense_config=None):
        """
        candidate_pool    how many seed-similar movies SVD re-ranks
        taste_pool        extra candidates pulled in from the user's taste profile
        for_you_pool      candidates considered for the seedless "For You" row
        cf_weight         weight of the SVD score vs. seed similarity (0..1)
        popularity_prior  Bayesian prior (in votes) used for the popularity score
        """
        started = time.perf_counter()
        self.candidate_pool = candidate_pool
        self.taste_pool = taste_pool
        self.for_you_pool = for_you_pool
        self.cf_weight = cf_weight

        movies_df = pd.DataFrame.from_records(
            Movie.objects.values("movie_id", "title", "year", "genres"),
            columns=["movie_id", "title", "year", "genres"],
        )
        if movies_df.empty:
            raise RuntimeError("No movies in the database. Run `python manage.py load_movielens` first.")

        ratings_df, self.defense_report = get_training_dataframe(defense_config)
        if ratings_df.empty:
            raise RuntimeError("No clean ratings available to train the collaborative model.")

        self.content = ContentEngine(movies_df)
        self.collab = CollaborativeEngine(ratings_df)
        self._build_popularity(ratings_df, popularity_prior)

        self.trained_at = datetime.now(timezone.utc)
        self.n_training_rows = len(ratings_df)
        logger.info(
            "Hybrid engine ready in %.1fs (%d movies, %d training rows, %d likes used)",
            time.perf_counter() - started, len(movies_df), len(ratings_df),
            self.defense_report.likes_used,
        )

    # ----- model statistics ------------------------------------------------

    def _build_popularity(self, ratings_df, prior):
        """Bayesian-average rating per movie, aligned with ContentEngine rows."""
        stats = ratings_df.groupby("movie_id")["rating"].agg(["count", "mean"])
        stats = stats.reindex(self.content.movie_ids).fillna({"count": 0, "mean": 0.0})
        global_mean = ratings_df["rating"].mean()
        votes = stats["count"].to_numpy()
        self.vote_counts = votes
        self.weighted_rating = (votes * stats["mean"].to_numpy() + prior * global_mean) / (votes + prior)

    def status(self):
        return {
            "trained_at": self.trained_at.isoformat(),
            "training_rows": self.n_training_rows,
            "likes_used": self.defense_report.likes_used,
            "likes_quarantined": self.defense_report.likes_quarantined,
        }

    # ----- user signals ----------------------------------------------------

    def taste_profile(self, user_id):
        """Live taste vector from likes (+1) and dislikes (-0.5). None if no usable
        signal, or if the user is currently restricted (quarantined)."""
        if is_user_restricted(user_id):
            return None
        likes = list(Like.objects.filter(user_id=user_id).values_list("movie_id", "value"))
        if not likes:
            return None

        rows, weights, genre_counts = [], [], Counter()
        n_likes = n_dislikes = 0
        for movie_id, value in likes:
            row = self.content.row_of.get(int(movie_id))
            if row is None:
                continue
            rows.append(row)
            if value > 0:
                weights.append(1.0)
                n_likes += 1
                genre_counts.update(self.content.genre_sets[row])
            else:
                weights.append(DISLIKE_WEIGHT)
                n_dislikes += 1
        if not rows:
            return None

        taste_vector = np.asarray(self.content.matrix[rows].T.dot(np.array(weights))).ravel()
        norm = np.linalg.norm(taste_vector)
        if norm == 0:
            return None
        sims = np.asarray(self.content.matrix.dot(taste_vector / norm)).ravel()
        return TasteProfile(np.round(sims, 6), n_likes, n_dislikes, genre_counts)

    def taste_summary(self, user_id):
        """JSON-friendly description of the user's likes, for the API."""
        counts = Counter(Like.objects.filter(user_id=user_id).values_list("value", flat=True))
        profile = self.taste_profile(user_id)
        top_genres = []
        if profile:
            top_genres = [
                {"genre": self.content.genre_label.get(g, g), "count": n}
                for g, n in profile.genre_counts.most_common(5)
            ]
        return {
            "n_likes": counts.get(Like.LIKE, 0),
            "n_dislikes": counts.get(Like.DISLIKE, 0),
            "active": profile is not None,
            "quarantined": is_user_restricted(user_id),
            "weight": round(profile.weight(), 3) if profile else 0.0,
            "top_genres": top_genres,
            "known_to_model": self.collab.knows_user(user_id),
        }

    def _excluded_for(self, user_id):
        """Movies the user has already rated, liked or disliked."""
        seen = set(Rating.objects.filter(user_id=user_id).values_list("movie_id", flat=True))
        seen |= set(Like.objects.filter(user_id=user_id).values_list("movie_id", flat=True))
        return seen

    def _allowed_mask(self, excluded):
        if not excluded:
            return np.ones(len(self.content.movie_ids), dtype=bool)
        return ~np.isin(self.content.movie_ids, np.fromiter(excluded, dtype=np.int64))

    def _top_rows(self, primary, allowed, limit, require_positive=True):
        """Row indices ranked by `primary` (ties broken by vote count), filtered by `allowed`."""
        order = np.lexsort((-self.vote_counts, -primary))
        rows = []
        for r in order:
            if not allowed[r] or (require_positive and primary[r] <= 0):
                continue
            rows.append(int(r))
            if len(rows) >= limit:
                break
        return rows

    def _predict(self, user_id, rows):
        if self.collab.knows_user(user_id):
            return self.collab.predict(user_id, self.content.movie_ids[rows]), "hybrid"
        return self.weighted_rating[rows], "cold_start"

    def _to_movies(self, rows, scores, similarities, predictions, taste_sims, strategy):
        ids = [int(self.content.movie_ids[r]) for r in rows]
        by_id = Movie.objects.in_bulk(ids)
        movies = []
        for mid, score, sim, pred, taste in zip(ids, scores, similarities, predictions, taste_sims):
            movie = by_id[mid]
            movie.score = round(float(score), 4)
            movie.similarity = round(float(sim), 4)
            movie.predicted_rating = round(float(pred), 2)
            movie.taste_match = round(float(taste), 4)
            movie.strategy = strategy
            movies.append(movie)
        return movies

    # ----- public API ------------------------------------------------------

    def recommend(self, user_id, movie_id, n=5):
        """Top-n movies for `user_id`, seeded by `movie_id` ("Because you watched...").

        score = w_cf * SVD prediction + w_seed * genre similarity to seed
                + w_taste * similarity to the user's like-based taste vector
        """
        sims = self.content.similarities(movie_id)
        taste = self.taste_profile(user_id)
        allowed = self._allowed_mask(self._excluded_for(user_id) | {int(movie_id)})

        candidates = self._top_rows(sims, allowed, self.candidate_pool)
        if taste is not None:
            chosen = set(candidates)
            candidates += [r for r in self._top_rows(taste.sims, allowed, self.taste_pool)
                           if r not in chosen]
        if not candidates:
            # Seed has no genres: fall back to the most popular unseen movies.
            candidates = self._top_rows(self.weighted_rating, allowed, self.candidate_pool,
                                        require_positive=False)
        candidates = np.array(candidates)
        cand_sims = sims[candidates]
        predictions, strategy = self._predict(user_id, candidates)

        w_taste = taste.weight() if taste is not None else 0.0
        w_cf = self.cf_weight * (1 - w_taste)
        w_seed = (1 - self.cf_weight) * (1 - w_taste)
        taste_sims = taste.sims[candidates] if taste is not None else np.zeros(len(candidates))
        scores = w_cf * _normalise_rating(predictions) + w_seed * cand_sims + w_taste * taste_sims

        top = np.argsort(-scores, kind="stable")[:n]
        movies = self._to_movies(candidates[top], scores[top], cand_sims[top],
                                 predictions[top], taste_sims[top], strategy)
        return Recommendations(
            movies=movies,
            strategy=strategy,
            weights={"collaborative": round(w_cf, 3), "seed_similarity": round(w_seed, 3),
                     "taste": round(w_taste, 3)},
            taste=self.taste_summary(user_id),
        )

    def for_you(self, user_id, n=10):
        """Seedless personal row driven by the user's likes, re-ranked by SVD.

        With likes: candidates = movies closest to the taste vector, score = 50% SVD + 50% taste.
        Without:    the user's top SVD predictions among popular movies (or plain popularity).
        """
        taste = self.taste_profile(user_id)
        allowed = self._allowed_mask(self._excluded_for(user_id))

        if taste is not None:
            candidates = self._top_rows(taste.sims, allowed, self.for_you_pool)
        else:
            candidates = self._top_rows(self.weighted_rating, allowed, self.for_you_pool,
                                        require_positive=False)
        candidates = np.array(candidates)
        predictions, base = self._predict(user_id, candidates)

        if taste is not None:
            taste_sims = taste.sims[candidates]
            scores = 0.5 * _normalise_rating(predictions) + 0.5 * taste_sims
            strategy = f"taste_{base}"
            weights = {"collaborative": 0.5, "taste": 0.5}
        else:
            taste_sims = np.zeros(len(candidates))
            scores = _normalise_rating(predictions)
            strategy = base if base == "hybrid" else "popular"
            weights = {"collaborative": 1.0, "taste": 0.0}

        top = np.argsort(-scores, kind="stable")[:n]
        movies = self._to_movies(candidates[top], scores[top], np.zeros(len(top)),
                                 predictions[top], taste_sims[top], strategy)
        return Recommendations(movies=movies, strategy=strategy, weights=weights,
                               taste=self.taste_summary(user_id))

    def similar_movies(self, movie_id, n=10):
        """Content-only neighbours of a movie (no user needed)."""
        sims = self.content.similarities(movie_id)
        seed_row = self.content.row_of[int(movie_id)]
        order = [r for r in np.lexsort((-self.vote_counts, -sims)) if r != seed_row][:n]
        zeros = np.zeros(len(order))
        return self._to_movies(order, sims[order], sims[order], self.weighted_rating[order],
                               zeros, "content")

    def popular_movies(self, n=20, genre=None):
        """Highest Bayesian-average movies, optionally restricted to one genre."""
        order = np.argsort(-self.weighted_rating, kind="stable")
        if genre:
            wanted = genre.strip().lower()
            order = [r for r in order if wanted in self.content.genre_sets[r]]
        order = list(order[:n])
        pops = self.weighted_rating[order]
        zeros = np.zeros(len(order))
        return self._to_movies(order, _normalise_rating(pops), zeros, pops, zeros, "popular")


# ---------------------------------------------------------------------------
# Process-wide engine with background retraining
# ---------------------------------------------------------------------------

_engine = None
_engine_lock = threading.Lock()
_state_lock = threading.Lock()
_pending_changes = 0
_retrain_thread = None


def get_engine():
    """Process-wide singleton; trained lazily on first call."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = HybridRecommender()
    return _engine


def reset_engine():
    """Drop the cached engine so the next call retrains (e.g. after loading new ratings)."""
    global _engine, _pending_changes
    with _engine_lock:
        _engine = None
    with _state_lock:
        _pending_changes = 0


def is_retraining():
    return _retrain_thread is not None and _retrain_thread.is_alive()


def _retrain_in_background():
    global _engine
    try:
        new_engine = HybridRecommender()
        with _engine_lock:
            _engine = new_engine
        logger.info("Background retrain complete")
    except Exception:
        logger.exception("Background retrain failed")
    finally:
        connection.close()


def note_training_data_changed():
    """Call after a trusted like changes. Starts a background retrain every
    `retrain_every()` changes (if one is not already running). Returns True if started."""
    global _pending_changes, _retrain_thread
    every = retrain_every()
    with _state_lock:
        _pending_changes += 1
        if every <= 0 or _pending_changes < every or is_retraining():
            return False
        _pending_changes = 0
        _retrain_thread = threading.Thread(
            target=_retrain_in_background, name="movify-retrain", daemon=True
        )
        _retrain_thread.start()
        return True


def engine_status():
    status = {
        "trained": _engine is not None,
        "retraining": is_retraining(),
        "pending_changes": _pending_changes,
        "retrain_every": retrain_every(),
    }
    if _engine is not None:
        status.update(_engine.status())
    return status


def get_hybrid_recommendations(user_id, movie_id, n=5):
    return get_engine().recommend(int(user_id), int(movie_id), n=n).movies
