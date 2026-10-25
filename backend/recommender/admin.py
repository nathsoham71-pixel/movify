from django.contrib import admin

from .models import Movie, Rating


@admin.register(Movie)
class MovieAdmin(admin.ModelAdmin):
    list_display = ("movie_id", "title", "year", "genres")
    search_fields = ("title", "genres")
    list_filter = ("year",)


@admin.register(Rating)
class RatingAdmin(admin.ModelAdmin):
    list_display = ("user_id", "movie", "rating", "timestamp")
    search_fields = ("movie__title",)
    list_filter = ("rating",)
    raw_id_fields = ("movie",)
