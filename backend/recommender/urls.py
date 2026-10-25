from django.urls import path

from . import views

app_name = "recommender"

urlpatterns = [
    path("health/", views.health, name="health"),
    path("recommendations/", views.recommendations, name="recommendations"),
    path("recommendations/for-you/", views.for_you, name="for-you"),
    path("movies/popular/", views.popular_movies, name="popular-movies"),
    path("movies/search/", views.search_movies, name="search-movies"),
    path("movies/<int:movie_id>/", views.movie_detail, name="movie-detail"),
    path("movies/<int:movie_id>/similar/", views.similar_movies, name="similar-movies"),
    path("genres/", views.genres, name="genres"),
    path("likes/", views.like_movie, name="like-movie"),
    path("users/<int:user_id>/profile/", views.user_profile, name="user-profile"),
    path("defense/report/", views.defense_report, name="defense-report"),
]
