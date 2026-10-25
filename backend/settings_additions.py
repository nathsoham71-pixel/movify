# ---- add to your Django settings.py -----------------------------------------
# Assumes BASE_DIR = cws/backend/server_side  (folder containing manage.py)
import os
from pathlib import Path

INSTALLED_APPS += ["movies"]

RECOMMENDER_DIR = BASE_DIR.parent / "recom_train"                    # where model.py lives
RECOMMENDER_MODEL_PATH = RECOMMENDER_DIR / "hybrid_model.pkl"       # created by: python model.py train
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")                    # optional, for posters

# ---- add to your PROJECT urls.py (the one next to settings.py) ---------------
# from django.urls import include, path
# urlpatterns = [ ..., path("", include("movies.urls")), ]
