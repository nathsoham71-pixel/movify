"""
Shilling attack (bot / push / nuke) defense.

Every detector below is a Django ORM aggregation that returns a set of suspicious
user IDs. Ratings from flagged users are excluded before the collaborative model
is trained, so fake profiles cannot push or nuke a movie's predicted score.

Detectors:
    volume      - more ratings than a person could plausibly give in total
    velocity    - too many ratings inside a single time window (scripted bursts)
    flat        - near-zero variance: every movie gets the same score
    extreme     - almost all ratings are the max (push) or min (nuke) value
    cluster     - several accounts give one movie the same extreme score in the
                  same time window (coordinated attack)

Usage:
    from recommender.bot_defense import get_clean_ratings_dataframe
    ratings_df = get_clean_ratings_dataframe()
"""
from dataclasses import dataclass, field

import pandas as pd
from django.db.models import BigIntegerField, Count, ExpressionWrapper, F, Q, StdDev

from .models import Rating

RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]


@dataclass(frozen=True)
class DefenseConfig:
    max_total_ratings: int = 1000
    velocity_window_seconds: int = 3600
    max_ratings_per_window: int = 500
    # Profile-shape checks are meaningless on tiny profiles.
    min_ratings_for_profile_checks: int = 20
    min_rating_stddev: float = 0.25
    max_extreme_share: float = 0.90
    max_rating_value: float = 5.0
    min_rating_value: float = 1.0
    cluster_window_seconds: int = 3600
    cluster_min_accounts: int = 3


@dataclass
class DefenseReport:
    total_users: int = 0
    total_ratings: int = 0
    kept_users: int = 0
    kept_ratings: int = 0
    flagged: dict = field(default_factory=dict)

    @property
    def flagged_users(self):
        return set().union(*self.flagged.values()) if self.flagged else set()

    def summary(self):
        lines = [
            f"Users:   {self.total_users:,} total, {len(self.flagged_users):,} flagged, "
            f"{self.kept_users:,} kept",
            f"Ratings: {self.total_ratings:,} total, "
            f"{self.total_ratings - self.kept_ratings:,} removed, {self.kept_ratings:,} kept",
        ]
        for reason, users in self.flagged.items():
            lines.append(f"  - {reason:<9} {len(users):>4} users")
        return "\n".join(lines)


def _time_bucket(seconds):
    return ExpressionWrapper(F("timestamp") / seconds, output_field=BigIntegerField())


def _extreme_ratings_q(config):
    return Q(rating__gte=config.max_rating_value) | Q(rating__lte=config.min_rating_value)


def detect_volume_abuse(queryset, config):
    return set(
        queryset.values("user_id")
        .annotate(n=Count("id"))
        .filter(n__gt=config.max_total_ratings)
        .values_list("user_id", flat=True)
    )


def detect_impossible_velocity(queryset, config):
    return set(
        queryset.annotate(bucket=_time_bucket(config.velocity_window_seconds))
        .values("user_id", "bucket")
        .annotate(n=Count("id"))
        .filter(n__gt=config.max_ratings_per_window)
        .values_list("user_id", flat=True)
    )


def detect_flat_profiles(queryset, config):
    return set(
        queryset.values("user_id")
        .annotate(n=Count("id"), spread=StdDev("rating"))
        .filter(n__gte=config.min_ratings_for_profile_checks, spread__lt=config.min_rating_stddev)
        .values_list("user_id", flat=True)
    )


def detect_extreme_bias(queryset, config):
    return set(
        queryset.values("user_id")
        .annotate(
            n=Count("id"),
            high=Count("id", filter=Q(rating__gte=config.max_rating_value)),
            low=Count("id", filter=Q(rating__lte=config.min_rating_value)),
        )
        .filter(n__gte=config.min_ratings_for_profile_checks)
        .filter(
            Q(high__gte=F("n") * config.max_extreme_share)
            | Q(low__gte=F("n") * config.max_extreme_share)
        )
        .values_list("user_id", flat=True)
    )


def detect_coordinated_clusters(queryset, config):
    window = config.cluster_window_seconds
    extreme = queryset.filter(_extreme_ratings_q(config))

    clusters = set(
        extreme.annotate(bucket=_time_bucket(window))
        .values("movie_id", "rating", "bucket")
        .annotate(accounts=Count("user_id", distinct=True))
        .filter(accounts__gte=config.cluster_min_accounts)
        .values_list("movie_id", "rating", "bucket")
    )
    if not clusters:
        return set()

    target_movies = {movie_id for movie_id, _, _ in clusters}
    candidates = extreme.filter(movie_id__in=target_movies).values_list(
        "user_id", "movie_id", "rating", "timestamp"
    )
    return {
        user_id
        for user_id, movie_id, rating, ts in candidates.iterator()
        if (movie_id, rating, ts // window) in clusters
    }


DETECTORS = {
    "volume": detect_volume_abuse,
    "velocity": detect_impossible_velocity,
    "flat": detect_flat_profiles,
    "extreme": detect_extreme_bias,
    "cluster": detect_coordinated_clusters,
}


def detect_shilling_accounts(config=None, queryset=None):
    """Return {reason: set(user_ids)} for every detector."""
    config = config or DefenseConfig()
    queryset = queryset if queryset is not None else Rating.objects.all()
    return {reason: detector(queryset, config) for reason, detector in DETECTORS.items()}


def run_bot_defense(config=None):
    """Return (clean_ratings_df, DefenseReport)."""
    config = config or DefenseConfig()
    all_ratings = Rating.objects.all()
    flagged = detect_shilling_accounts(config, all_ratings)

    report = DefenseReport(
        total_users=all_ratings.values("user_id").distinct().count(),
        total_ratings=all_ratings.count(),
        flagged=flagged,
    )

    clean = all_ratings.exclude(user_id__in=report.flagged_users)
    df = pd.DataFrame.from_records(
        clean.values_list(*RATING_COLUMNS).iterator(chunk_size=10_000),
        columns=RATING_COLUMNS,
    ).astype({"user_id": "int64", "movie_id": "int64", "rating": "float64", "timestamp": "int64"})

    report.kept_ratings = len(df)
    report.kept_users = df["user_id"].nunique()
    return df, report


def get_clean_ratings_dataframe(config=None):
    """DataFrame[user_id, movie_id, rating, timestamp] with shilling accounts removed."""
    df, _ = run_bot_defense(config)
    return df
