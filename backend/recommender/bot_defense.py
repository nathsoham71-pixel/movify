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

Live defense (second half of this module) watches like/dislike clicks as they
happen and escalates trusted -> warned -> restricted. Restricted users' likes are
quarantined: they are excluded from model training until the restriction lifts.

Usage:
    from recommender.bot_defense import get_clean_ratings_dataframe, get_training_dataframe
    ratings_df = get_clean_ratings_dataframe()          # MovieLens ratings only
    train_df, report = get_training_dataframe()         # ratings + trusted likes
"""
from dataclasses import asdict, dataclass, field
from datetime import timedelta

import pandas as pd
from django.db.models import BigIntegerField, Count, ExpressionWrapper, F, Q, StdDev
from django.utils import timezone

from .models import Like, LikeEvent, Rating, UserTrust

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
    likes_used: int = 0
    likes_quarantined: int = 0

    @property
    def flagged_users(self):
        return set().union(*self.flagged.values()) if self.flagged else set()

    def summary(self):
        lines = [
            f"Users:   {self.total_users:,} total, {len(self.flagged_users):,} flagged, "
            f"{self.kept_users:,} kept",
            f"Ratings: {self.total_ratings:,} total, "
            f"{self.total_ratings - self.kept_ratings:,} removed, {self.kept_ratings:,} kept",
            f"Likes:   {self.likes_used:,} used, {self.likes_quarantined:,} quarantined",
        ]
        for reason, users in self.flagged.items():
            lines.append(f"  - {reason:<9} {len(users):>4} users")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "total_users": self.total_users,
            "total_ratings": self.total_ratings,
            "kept_users": self.kept_users,
            "kept_ratings": self.kept_ratings,
            "removed_ratings": self.total_ratings - self.kept_ratings,
            "likes_used": self.likes_used,
            "likes_quarantined": self.likes_quarantined,
            "flagged": {reason: sorted(users) for reason, users in self.flagged.items()},
        }


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


def get_training_dataframe(config=None):
    """Clean MovieLens ratings plus likes from trusted users, as one DataFrame.

    A like becomes a 5.0 rating and a dislike a 1.0 rating. Likes from users
    flagged by the batch detectors or currently restricted by the live defense
    are quarantined (left out). A like overrides an older MovieLens rating for
    the same user/movie pair.
    """
    df, report = run_bot_defense(config)
    excluded = report.flagged_users | restricted_user_ids()

    likes = Like.objects.values_list("user_id", "movie_id", "value", "updated_at")
    rows, quarantined = [], 0
    for user_id, movie_id, value, updated_at in likes:
        if user_id in excluded:
            quarantined += 1
            continue
        rows.append((user_id, movie_id, Like.IMPLIED_RATING[value], int(updated_at.timestamp())))

    report.likes_used = len(rows)
    report.likes_quarantined = quarantined
    if rows:
        likes_df = pd.DataFrame(rows, columns=RATING_COLUMNS).astype(df.dtypes.to_dict())
        df = (
            pd.concat([df, likes_df], ignore_index=True)
            .drop_duplicates(subset=["user_id", "movie_id"], keep="last")
            .reset_index(drop=True)
        )
    return df, report


# ---------------------------------------------------------------------------
# Live defense: evaluates like/dislike activity as it happens
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveDefenseConfig:
    window_seconds: int = 60
    warn_actions_per_window: int = 8        # > this in one window -> warning + strike
    restrict_actions_per_window: int = 15   # > this in one window -> immediate restriction
    same_movie_actions_per_window: int = 4  # toggling one movie this often -> warning
    strikes_before_restriction: int = 3
    restriction_minutes: int = 10
    cluster_window_seconds: int = 600       # coordinated push: fresh accounts hitting one movie
    cluster_min_accounts: int = 3
    cluster_max_other_actions: int = 2      # "fresh" = no ratings and <= this many other clicks

    def to_dict(self):
        return asdict(self)


@dataclass
class LiveAssessment:
    user_id: int
    status: str
    allowed: bool
    message: str = ""
    reasons: list = field(default_factory=list)
    actions_in_window: int = 0
    strikes: int = 0
    restricted_until: object = None
    config: LiveDefenseConfig = field(default_factory=LiveDefenseConfig)

    def to_dict(self):
        return {
            "status": self.status,
            "allowed": self.allowed,
            "message": self.message,
            "reasons": self.reasons,
            "actions_in_window": self.actions_in_window,
            "strikes": self.strikes,
            "strikes_before_restriction": self.config.strikes_before_restriction,
            "restricted_until": self.restricted_until.isoformat() if self.restricted_until else None,
            "limits": {
                "window_seconds": self.config.window_seconds,
                "warn_after": self.config.warn_actions_per_window,
                "restrict_after": self.config.restrict_actions_per_window,
                "same_movie_limit": self.config.same_movie_actions_per_window,
            },
        }


def _load_trust(user_id, now):
    """Fetch (or create) the trust row, moving expired restrictions to probation."""
    trust, _ = UserTrust.objects.get_or_create(user_id=user_id)
    if trust.status == UserTrust.RESTRICTED and not trust.is_restricted(now):
        trust.status = UserTrust.WARNED
        trust.save(update_fields=["status", "updated_at"])
    return trust


def is_user_restricted(user_id, now=None):
    trust = UserTrust.objects.filter(user_id=user_id).first()
    return bool(trust and trust.is_restricted(now))


def restricted_user_ids(now=None):
    now = now or timezone.now()
    return set(
        UserTrust.objects.filter(status=UserTrust.RESTRICTED, restricted_until__gt=now)
        .values_list("user_id", flat=True)
    )


def _actions_in_window(user_id, since):
    return LikeEvent.objects.filter(user_id=user_id, created_at__gte=since).count()


def _coordinated_push_size(user_id, movie_id, now, config):
    """Number of fresh accounts (incl. this one) that hit `movie_id` in the cluster window,
    or 0 if this does not look like a coordinated attack."""
    since = now - timedelta(seconds=config.cluster_window_seconds)
    recent_users = set(
        LikeEvent.objects.filter(movie_id=movie_id, created_at__gte=since)
        .exclude(action="remove")
        .values_list("user_id", flat=True)
        .distinct()
    )
    if user_id not in recent_users or len(recent_users) < config.cluster_min_accounts:
        return 0

    with_history = set(
        Rating.objects.filter(user_id__in=recent_users).values_list("user_id", flat=True).distinct()
    )
    other_actions = dict(
        LikeEvent.objects.filter(user_id__in=recent_users)
        .exclude(movie_id=movie_id)
        .values_list("user_id")
        .annotate(n=Count("id"))
        .values_list("user_id", "n")
    )
    fresh = {
        uid for uid in recent_users
        if uid not in with_history and other_actions.get(uid, 0) <= config.cluster_max_other_actions
    }
    if user_id not in fresh or len(fresh) < config.cluster_min_accounts:
        return 0
    return len(fresh)


def describe_trust(user_id, config=None, now=None):
    """Current trust status for a user, without recording any new activity."""
    config = config or LiveDefenseConfig()
    now = now or timezone.now()
    trust = _load_trust(user_id, now)
    restricted = trust.is_restricted(now)
    message = ""
    if restricted:
        message = (f"Your account is restricted until {trust.restricted_until:%H:%M} because of "
                   f"suspicious rating activity ({trust.last_reason}). Likes are not being accepted "
                   "and your recent likes are excluded from the recommendation model.")
    elif trust.status == UserTrust.WARNED:
        message = (f"Warning {trust.strikes}/{config.strikes_before_restriction} on record: "
                   f"{trust.last_reason}. Keep your activity natural to avoid a restriction.")
    return LiveAssessment(
        user_id=user_id,
        status=trust.status,
        allowed=not restricted,
        message=message,
        reasons=[trust.last_reason] if trust.last_reason else [],
        actions_in_window=_actions_in_window(user_id, now - timedelta(seconds=config.window_seconds)),
        strikes=trust.strikes,
        restricted_until=trust.restricted_until if restricted else None,
        config=config,
    )


def assess_live_activity(user_id, movie_id, config=None, now=None):
    """Evaluate a user's recent activity right after a like/dislike click was logged.

    Updates the user's `UserTrust` row and returns a `LiveAssessment` whose
    `allowed` flag says whether the click should take effect.
    """
    config = config or LiveDefenseConfig()
    now = now or timezone.now()
    trust = _load_trust(user_id, now)
    window_start = now - timedelta(seconds=config.window_seconds)

    if trust.is_restricted(now):
        verdict = describe_trust(user_id, config, now)
        verdict.allowed = False
        return verdict

    events = LikeEvent.objects.filter(user_id=user_id, created_at__gte=window_start)
    actions = events.count()
    same_movie = events.filter(movie_id=movie_id).count()

    reasons, severe = [], False
    if actions > config.restrict_actions_per_window:
        reasons.append(f"{actions} actions in {config.window_seconds}s "
                       f"(hard limit {config.restrict_actions_per_window})")
        severe = True
    elif actions > config.warn_actions_per_window:
        reasons.append(f"{actions} actions in {config.window_seconds}s "
                       f"(limit {config.warn_actions_per_window})")
    if same_movie >= config.same_movie_actions_per_window:
        reasons.append(f"same movie toggled {same_movie} times in {config.window_seconds}s")
    cluster = _coordinated_push_size(user_id, movie_id, now, config)
    if cluster:
        reasons.append(f"{cluster} new accounts targeted this movie within "
                       f"{config.cluster_window_seconds // 60} min")

    if not reasons:
        return LiveAssessment(
            user_id=user_id, status=trust.status, allowed=True,
            actions_in_window=actions, strikes=trust.strikes, config=config,
        )

    reason_text = "; ".join(reasons)
    # One strike per window, so a burst of clicks is a single offence.
    if trust.flagged_at is None or trust.flagged_at < window_start:
        trust.strikes += 1
    trust.flagged_at = now
    trust.last_reason = reason_text[:255]
    trust.warnings_issued += 1

    if severe or trust.strikes >= config.strikes_before_restriction:
        trust.status = UserTrust.RESTRICTED
        trust.restricted_until = now + timedelta(minutes=config.restriction_minutes)
        allowed = False
        message = (f"Suspicious rating activity detected: {reason_text}. Your account is "
                   f"restricted for {config.restriction_minutes} minutes. Likes are rejected and "
                   "your recent likes are excluded from the recommendation model until then.")
    else:
        trust.status = UserTrust.WARNED
        allowed = True
        message = (f"Warning {trust.strikes}/{config.strikes_before_restriction}: {reason_text}. "
                   "Slow down. Continued activity like this will restrict your account.")
    trust.save()

    return LiveAssessment(
        user_id=user_id, status=trust.status, allowed=allowed, message=message,
        reasons=reasons, actions_in_window=actions, strikes=trust.strikes,
        restricted_until=trust.restricted_until if trust.status == UserTrust.RESTRICTED else None,
        config=config,
    )


def live_defense_overview(config=None, now=None):
    """Every user the live defense has ever flagged, for the defense report endpoint."""
    config = config or LiveDefenseConfig()
    now = now or timezone.now()
    users = []
    for trust in UserTrust.objects.exclude(status=UserTrust.TRUSTED, strikes=0).order_by("-updated_at"):
        users.append({
            "user_id": trust.user_id,
            "status": UserTrust.RESTRICTED if trust.is_restricted(now) else (
                UserTrust.WARNED if trust.status != UserTrust.TRUSTED else UserTrust.TRUSTED),
            "strikes": trust.strikes,
            "warnings_issued": trust.warnings_issued,
            "last_reason": trust.last_reason,
            "restricted_until": trust.restricted_until.isoformat()
            if trust.is_restricted(now) else None,
        })
    return {"config": config.to_dict(), "users": users}
