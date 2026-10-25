from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


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
