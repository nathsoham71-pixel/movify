from django.db import models


class MovieArtwork(models.Model):
    """
    Cache of poster / backdrop / overview fetched from TMDB.

    MovieLens only gives us ids (links.csv -> tmdbId), not images, so the first time a movie
    is recommended we ask TMDB once and store the result here. After that, no network calls.
    """

    movie_id = models.IntegerField(primary_key=True)          # MovieLens movieId
    tmdb_id = models.IntegerField(null=True, blank=True)
    poster_path = models.CharField(max_length=255, blank=True, default="")
    backdrop_path = models.CharField(max_length=255, blank=True, default="")
    overview = models.TextField(blank=True, default="")
    fetched_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"MovieArtwork({self.movie_id})"
