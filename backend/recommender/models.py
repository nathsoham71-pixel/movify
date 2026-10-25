from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone


class Movie(models.Model):
    """A movie from the MovieLens `movies.csv` file."""

    # MovieLens IDs are used as primary keys so ratings.csv can reference them directly.
    movie_id = models.PositiveIntegerField(primary_key=True)
    title = models.CharField(max_length=255, db_index=True)
    year = models.PositiveSmallIntegerField(null=True, blank=True)
    # Pipe-separated, e.g. "Adventure|Animation|Children".
    genres = models.CharField(max_length=255, blank=True, default="")

    class Meta:
        ordering = ["movie_id"]

    def __str__(self):
        return self.title

    @property
    def genre_list(self):
        if not self.genres or self.genres == "(no genres listed)":
            return []
        return self.genres.split("|")


class Rating(models.Model):
    """A single user rating from the MovieLens `ratings.csv` file."""

    # MovieLens users are anonymous integers, not Django auth users.
    user_id = models.PositiveIntegerField(db_index=True)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name="ratings")
    rating = models.FloatField(
        validators=[MinValueValidator(0.5), MaxValueValidator(5.0)],
    )
    # Unix epoch seconds, kept as an integer for fast velocity calculations.
    timestamp = models.PositiveBigIntegerField()

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["user_id", "movie"], name="unique_user_movie_rating"),
        ]
        indexes = [
            models.Index(fields=["user_id", "timestamp"]),
        ]

    def __str__(self):
        return f"User {self.user_id} rated {self.movie_id}: {self.rating}"


class Like(models.Model):
    """Current like/dislike state of a user for a movie (set through the UI)."""

    LIKE = 1
    DISLIKE = -1
    VALUE_CHOICES = [(LIKE, "Like"), (DISLIKE, "Dislike")]

    # Rating value a like/dislike is converted to when the SVD model retrains.
    IMPLIED_RATING = {LIKE: 5.0, DISLIKE: 1.0}

    user_id = models.PositiveIntegerField(db_index=True)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name="likes")
    value = models.SmallIntegerField(choices=VALUE_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["user_id", "movie"], name="unique_user_movie_like"),
        ]

    def __str__(self):
        return f"User {self.user_id} {self.get_value_display().lower()}d {self.movie_id}"

    @property
    def implied_rating(self):
        return self.IMPLIED_RATING[self.value]


class LikeEvent(models.Model):
    """Append-only log of every like/dislike/remove click. Toggling deletes `Like`
    rows, so velocity and spam detection must look at events, not current state."""

    ACTION_CHOICES = [("like", "Like"), ("dislike", "Dislike"), ("remove", "Remove")]

    user_id = models.PositiveIntegerField(db_index=True)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name="like_events")
    action = models.CharField(max_length=8, choices=ACTION_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        indexes = [
            models.Index(fields=["user_id", "created_at"]),
            models.Index(fields=["movie", "created_at"]),
        ]

    def __str__(self):
        return f"{self.created_at:%H:%M:%S} user {self.user_id} {self.action} {self.movie_id}"


class UserTrust(models.Model):
    """Live shilling-defense status for a user."""

    TRUSTED = "trusted"
    WARNED = "warned"
    RESTRICTED = "restricted"
    STATUS_CHOICES = [(TRUSTED, "Trusted"), (WARNED, "Warned"), (RESTRICTED, "Restricted")]

    user_id = models.PositiveIntegerField(primary_key=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default=TRUSTED)
    strikes = models.PositiveSmallIntegerField(default=0)
    warnings_issued = models.PositiveIntegerField(default=0)
    last_reason = models.CharField(max_length=255, blank=True, default="")
    flagged_at = models.DateTimeField(null=True, blank=True)
    restricted_until = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "user trust"

    def __str__(self):
        return f"User {self.user_id}: {self.status} ({self.strikes} strikes)"

    def is_restricted(self, now=None):
        now = now or timezone.now()
        return (
            self.status == self.RESTRICTED
            and self.restricted_until is not None
            and self.restricted_until > now
        )
