from django.contrib import admin

from .models import Like, LikeEvent, Movie, Rating, UserTrust


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


@admin.register(Like)
class LikeAdmin(admin.ModelAdmin):
    list_display = ("user_id", "movie", "value", "updated_at")
    list_filter = ("value",)
    search_fields = ("movie__title",)
    raw_id_fields = ("movie",)


@admin.register(LikeEvent)
class LikeEventAdmin(admin.ModelAdmin):
    list_display = ("created_at", "user_id", "action", "movie")
    list_filter = ("action",)
    raw_id_fields = ("movie",)


@admin.register(UserTrust)
class UserTrustAdmin(admin.ModelAdmin):
    list_display = ("user_id", "status", "strikes", "warnings_issued", "restricted_until", "last_reason")
    list_filter = ("status",)
