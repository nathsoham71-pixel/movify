from collections import Counter

import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from .artwork import attach_artwork
from .recommender_service import get_model

ROW_SIZE = 20        # movies per carousel
POOL_SIZE = 150      # recommendations fetched once, then sliced into rows
GENRE_ROWS = 3
HIDDEN_GENRES = {"(no genres listed)", "IMAX"}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_liked(raw):
    """'1:5,356:4.5' -> {1: 5.0, 356: 4.5}  (lets a brand-new user get cold-start picks)."""
    liked = {}
    for part in (raw or "").split(","):
        try:
            movie_id, rating = part.split(":")
            liked[int(movie_id)] = float(rating)
        except ValueError:
            continue
    return liked


def _genres(raw):
    return [g for g in str(raw).split("|") if g and g not in HIDDEN_GENRES]


def _records(df):
    """DataFrame returned by the model -> list of JSON-ready movie dicts."""
    movies = []
    for row in df.to_dict("records"):
        tmdb, imdb = row.get("tmdbId"), row.get("imdbId")
        movies.append(
            {
                "id": int(row["movieId"]),
                "title": row["title"],
                "genres": _genres(row["genres"]),
                "tmdb_id": int(float(tmdb)) if pd.notna(tmdb) else None,
                "imdb_id": str(imdb) if pd.notna(imdb) else None,
                "poster_url": None,       # filled by attach_artwork()
                "backdrop_url": None,
                "overview": "",
            }
        )
    return movies


# ----------------------------------------------------------------------------
# views
# ----------------------------------------------------------------------------
def index(request):
    """Serves the single-page frontend; all movie data is loaded by app.js via fetch()."""
    return render(request, "movies/index.html")


@require_GET
def recommendations_api(request):
    """
    GET /api/recommendations/?user_id=5
    GET /api/recommendations/?liked=1:5,356:4.5      (new user, no account yet)

    Response:
      { "hero": {movie},
        "rows": [ {"id": "top_picks", "title": "...", "movies": [ {movie}, ... ]}, ... ] }
    """
    user_id = _int_or_none(request.GET.get("user_id"))
    liked = _parse_liked(request.GET.get("liked"))

    model = get_model()

    # =========================================================================
    #  >>> YOUR RECOMMENDATION MODEL IS CALLED HERE <<<
    #  HybridRecommender.recommend(user_id, n, liked) returns a DataFrame with columns:
    #  movieId, title, genres, score, imdbId, tmdbId
    #  (bot-flagged users are already excluded from training inside the model itself)
    # =========================================================================
    pool = model.recommend(user_id=user_id, n=POOL_SIZE, liked=liked)

    personalised = bool(liked) or user_id in model.user_hist
    top_df, rest_df = pool.head(ROW_SIZE), pool.iloc[ROW_SIZE:]

    rows = [
        {
            "id": "top_picks",
            "title": "Top Picks for You" if personalised else "Popular Right Now",
            "movies": _records(top_df),
        }
    ]

    # "Because you liked X" -> content-based neighbours of the user's best-rated movie
    history = {**model.user_hist.get(user_id, {}), **liked}
    if history:
        seed_id = max(history, key=history.get)
        if seed_id in model.pos_of:
            seed_title = model.movies.iloc[model.pos_of[seed_id]]["title"]
            similar = model.similar_movies(seed_id, n=ROW_SIZE)
            if model.links is not None:
                similar["tmdbId"] = similar.movieId.map(model.links.tmdbId)
                similar["imdbId"] = similar.movieId.map(model.links.imdbId)
            rows.append(
                {"id": "because", "title": f"Because You Liked {seed_title}", "movies": _records(similar)}
            )

    # Genre rows built from the rest of the ranked pool (most common genres first)
    counts = Counter(g for raw in rest_df.genres for g in _genres(raw))
    for i, (genre, _) in enumerate(counts.most_common(GENRE_ROWS), start=1):
        subset = rest_df[rest_df.genres.apply(lambda s: genre in s.split("|"))].head(ROW_SIZE)
        rows.append({"id": f"genre_{i}", "title": f"{genre} Picks", "movies": _records(subset)})

    # One artwork lookup for every movie on the page (cached in the DB after the first time)
    all_movies = [m for row in rows for m in row["movies"]]
    attach_artwork(all_movies)

    top_movies = rows[0]["movies"]
    hero = next((m for m in top_movies if m["backdrop_url"]), top_movies[0] if top_movies else None)

    return JsonResponse({"hero": hero, "rows": rows})
