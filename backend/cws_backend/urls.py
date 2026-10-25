"""URL configuration for the cws_backend project."""
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("recommender.urls")),
]
