from django.urls import path

from . import views

app_name = "movies"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/recommendations/", views.recommendations_api, name="recommendations"),
]
