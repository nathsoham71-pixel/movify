"""
Hybrid movie recommendation engine.

    Content engine        TF-IDF over movie genres + cosine similarity.
                          Works for any movie, even with zero ratings (cold start).
    Collaborative engine  Surprise SVD matrix factorisation, trained only on the
                          sanitised ratings returned by bot_defense.
    Hybrid                Candidates come from the content engine (movies similar
                          to the seed), then each is re-ranked by the user's SVD
                          predicted rating blended with its genre similarity.

Usage:
    from recommender.ml_engine import get_hybrid_recommendations
    movies = get_hybrid_recommendations(user_id=1, movie_id=10)   # list[Movie]

The engine is trained once per process on first use (a few seconds) and then
served from memory.
"""
import logging
import threading
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

from .bot_defense import run_bot_defense
from .models import Movie, Rating

logger = logging.getLogger(__name__)

NO_GENRES = "(no genres listed)"
RATING_SCALE = (0.5, 5.0)


class MovieNotFound(LookupError):
    pass


def _genre_tokens(genres):
    """'Sci-Fi|Film-Noir' -> ['sci-fi', 'film-noir'] (keeps multi-word genres intact)."""
    if not genres or genres == NO_GENRES:
        return []
    return [g.strip().lower() for g in genres.split("|") if g.strip()]


class ContentEngine:
    """TF-IDF genre vectors. Similarities are computed per query, never as a full NxN matrix."""

    def __init__(self, movies_df):
        self.movie_ids = movies_df["movie_id"].to_numpy()
        self.row_of = {int(mid): row for row, mid in enumerate(self.movie_ids)}
        self.genre_sets = [set(_genre_tokens(g)) for g in movies_df["genres"].fillna("")]
        self.vectorizer = TfidfVectorizer(analyzer=_genre_tokens)
        self.matrix = self.vectorizer.fit_transform(movies_df["genres"].fillna(""))

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


class HybridRecommender:
    def __init__(self, candidate_pool=50, cf_weight=0.7, popularity_prior=20, defense_config=None):
        """
        candidate_pool    how many content-similar movies SVD re-ranks
        cf_weight         weight of the SVD score vs. genre similarity (0..1)
        popularity_prior  Bayesian prior (in votes) used for the popularity score
        """
        started = time.perf_counter()
        self.candidate_pool = candidate_pool
        self.cf_weight = cf_weight

        movies_df = pd.DataFrame.from_records(
            Movie.objects.values("movie_id", "title", "year", "genres"),
            columns=["movie_id", "title", "year", "genres"],
        )
        if movies_df.empty:
            raise RuntimeError("No movies in the database. Run `python manage.py load_movielens` first.")

        ratings_df, self.defense_report = run_bot_defense(defense_config)
        if ratings_df.empty:
            raise RuntimeError("No clean ratings available to train the collaborative model.")

        self.content = ContentEngine(movies_df)
        self.collab = CollaborativeEngine(ratings_df)
        self._build_popularity(ratings_df, popularity_prior)

        logger.info(
            "Hybrid engine ready in %.1fs (%d movies, %d clean ratings)",
            time.perf_counter() - started, len(movies_df), len(ratings_df),
        )

    def _build_popularity(self, ratings_df, prior):
        """Bayesian-average rating per movie, aligned with ContentEngine rows."""
        stats = ratings_df.groupby("movie_id")["rating"].agg(["count", "mean"])
        stats = stats.reindex(self.content.movie_ids).fillna({"count": 0, "mean": 0.0})
        global_mean = ratings_df["rating"].mean()
        votes = stats["count"].to_numpy()
        self.vote_counts = votes
        self.weighted_rating = (votes * stats["mean"].to_numpy() + prior * global_mean) / (votes + prior)

    @staticmethod
    def _normalise_rating(values):
        low, high = RATING_SCALE
        return (np.asarray(values) - low) / (high - low)

    @staticmethod
    def _seen_movie_ids(user_id):
        return set(Rating.objects.filter(user_id=user_id).values_list("movie_id", flat=True))

    def _to_movies(self, rows, scores, similarities, predictions, strategy):
        ids = [int(self.content.movie_ids[r]) for r in rows]
        by_id = Movie.objects.in_bulk(ids)
        movies = []
        for mid, score, sim, pred in zip(ids, scores, similarities, predictions):
            movie = by_id[mid]
            movie.score = round(float(score), 4)
            movie.similarity = round(float(sim), 4)
            movie.predicted_rating = round(float(pred), 2)
            movie.strategy = strategy
            movies.append(movie)
        return movies

    def recommend(self, user_id, movie_id, n=5):
        """Top-n Movie objects for `user_id`, seeded by `movie_id`.

        Each returned Movie has extra attributes: score, similarity,
        predicted_rating and strategy ("hybrid" or "cold_start").
        """
        sims = self.content.similarities(movie_id)
        exclude = self._seen_movie_ids(user_id) | {int(movie_id)}
        allowed = np.array([int(mid) not in exclude for mid in self.content.movie_ids])

        # Primary key: genre similarity. Tie-breaker: number of ratings.
        order = np.lexsort((-self.vote_counts, -sims))
        candidates = [r for r in order if allowed[r] and sims[r] > 0][: self.candidate_pool]
        if not candidates:
            # Seed has no genres: fall back to the most popular unseen movies.
            order = np.argsort(-self.weighted_rating)
            candidates = [r for r in order if allowed[r]][: self.candidate_pool]
        candidates = np.array(candidates)
        cand_sims = sims[candidates]

        if self.collab.knows_user(user_id):
            strategy = "hybrid"
            predictions = self.collab.predict(user_id, self.content.movie_ids[candidates])
        else:
            strategy = "cold_start"
            predictions = self.weighted_rating[candidates]

        scores = self.cf_weight * self._normalise_rating(predictions) + (1 - self.cf_weight) * cand_sims
        top = np.argsort(-scores, kind="stable")[:n]
        return self._to_movies(candidates[top], scores[top], cand_sims[top], predictions[top], strategy)

    def similar_movies(self, movie_id, n=10):
        """Content-only neighbours of a movie (no user needed)."""
        sims = self.content.similarities(movie_id)
        seed_row = self.content.row_of[int(movie_id)]
        order = [r for r in np.lexsort((-self.vote_counts, -sims)) if r != seed_row][:n]
        return self._to_movies(order, sims[order], sims[order], self.weighted_rating[order], "content")

    def popular_movies(self, n=20, genre=None):
        """Highest Bayesian-average movies, optionally restricted to one genre."""
        order = np.argsort(-self.weighted_rating, kind="stable")
        if genre:
            wanted = genre.strip().lower()
            order = [r for r in order if wanted in self.content.genre_sets[r]]
        order = list(order[:n])
        pops = self.weighted_rating[order]
        return self._to_movies(order, self._normalise_rating(pops), np.zeros(len(order)), pops, "popular")


_engine = None
_engine_lock = threading.Lock()


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
    global _engine
    with _engine_lock:
        _engine = None


def get_hybrid_recommendations(user_id, movie_id, n=5):
    return get_engine().recommend(int(user_id), int(movie_id), n=n)
