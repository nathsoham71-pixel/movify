"""
JSON REST API for the movie recommender.

    GET  /api/health/
    GET  /api/recommendations/?user_id=1&movie_id=10[&n=5]
    GET  /api/recommendations/for-you/?user_id=1[&n=10]
    GET  /api/movies/popular/[?n=20&genre=Comedy]
    GET  /api/movies/search/?q=toy[&n=10]
    GET  /api/movies/<movie_id>/
    GET  /api/movies/<movie_id>/similar/[?n=10]
    GET  /api/genres/
    POST /api/likes/            {"user_id": 1, "movie_id": 10, "value": 1 | -1 | 0}
    GET  /api/users/<user_id>/profile/
    GET  /api/defense/report/
"""
import json
import logging
from functools import wraps

from django.db import transaction
from django.db.models import Count
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .bot_defense import assess_live_activity, describe_trust, live_defense_overview
from .ml_engine import MovieNotFound, engine_status, get_engine, note_training_data_changed
from .models import Like, LikeEvent, Movie, Rating, UserTrust

logger = logging.getLogger(__name__)

MAX_RESULTS = 50
LIKE_ACTIONS = {Like.LIKE: "like", Like.DISLIKE: "dislike", 0: "remove"}


class BadRequest(ValueError):
    pass


def error(message, status):
    return JsonResponse({"error": message}, status=status)


def api_view(view=None, *, methods=("GET",)):
    """JSON view with uniform error responses. Views return a dict (-> 200) or an HttpResponse."""
    def decorate(func):
        # Read-only or stateless JSON API: exempt from CSRF so non-allowed methods get a 405.
        @csrf_exempt
        @require_http_methods(list(methods))
        @wraps(func)
        def wrapper(request, *args, **kwargs):
            try:
                result = func(request, *args, **kwargs)
                return result if isinstance(result, HttpResponse) else JsonResponse(result)
            except BadRequest as exc:
                return error(str(exc), 400)
            except (MovieNotFound, Movie.DoesNotExist):
                return error("Movie not found.", 404)
            except RuntimeError as exc:
                logger.exception("Recommendation engine unavailable")
                return error(str(exc), 503)
        return wrapper

    return decorate(view) if view is not None else decorate


def _to_int(raw, name, minimum=1, maximum=None):
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise BadRequest(f"'{name}' must be an integer.") from None
    if value < minimum:
        raise BadRequest(f"'{name}' must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise BadRequest(f"'{name}' must be <= {maximum}.")
    return value


def int_param(request, name, default=None, minimum=1, maximum=None):
    raw = request.GET.get(name)
    if raw in (None, ""):
        if default is None:
            raise BadRequest(f"Missing required query parameter '{name}'.")
        return default
    return _to_int(raw, name, minimum, maximum)


def json_body(request):
    if not request.body:
        return {}
    try:
        data = json.loads(request.body)
    except ValueError:
        raise BadRequest("Request body must be valid JSON.") from None
    if not isinstance(data, dict):
        raise BadRequest("Request body must be a JSON object.")
    return data


def body_int(body, name, minimum=1, maximum=None):
    if name not in body:
        raise BadRequest(f"Missing required field '{name}'.")
    return _to_int(body[name], name, minimum, maximum)


def serialize_movie(movie):
    data = {
        "movie_id": movie.movie_id,
        "title": movie.title,
        "year": movie.year,
        "genres": movie.genre_list,
    }
    for attr in ("score", "similarity", "predicted_rating", "taste_match", "strategy"):
        if hasattr(movie, attr):
            data[attr] = getattr(movie, attr)
    return data


def serialize_recommendations(result):
    return {
        "strategy": result.strategy,
        "weights": result.weights,
        "taste": result.taste,
        "count": len(result.movies),
        "results": [serialize_movie(m) for m in result.movies],
    }


# ----- read endpoints ------------------------------------------------------


@api_view
def health(request):
    return {
        "status": "ok",
        "movies": Movie.objects.count(),
        "ratings": Rating.objects.count(),
        "likes": Like.objects.count(),
        "engine": engine_status(),
    }


@api_view
def recommendations(request):
    user_id = int_param(request, "user_id")
    movie_id = int_param(request, "movie_id")
    n = int_param(request, "n", default=5, maximum=MAX_RESULTS)

    seed = Movie.objects.get(movie_id=movie_id)
    result = get_engine().recommend(user_id, movie_id, n=n)
    return {"user_id": user_id, "seed_movie": serialize_movie(seed),
            **serialize_recommendations(result)}


@api_view
def for_you(request):
    user_id = int_param(request, "user_id")
    n = int_param(request, "n", default=10, maximum=MAX_RESULTS)
    result = get_engine().for_you(user_id, n=n)
    return {"user_id": user_id, **serialize_recommendations(result)}


@api_view
def popular_movies(request):
    n = int_param(request, "n", default=20, maximum=MAX_RESULTS)
    genre = request.GET.get("genre", "").strip() or None
    results = get_engine().popular_movies(n=n, genre=genre)
    return {"genre": genre, "count": len(results), "results": [serialize_movie(m) for m in results]}


@api_view
def search_movies(request):
    query = request.GET.get("q", "").strip()
    if len(query) < 2:
        raise BadRequest("Query parameter 'q' must be at least 2 characters.")
    n = int_param(request, "n", default=10, maximum=MAX_RESULTS)
    movies = (
        Movie.objects.filter(title__icontains=query)
        .annotate(num_ratings=Count("ratings"))
        .order_by("-num_ratings", "title")[:n]
    )
    return {"query": query, "count": len(movies), "results": [serialize_movie(m) for m in movies]}


@api_view
def movie_detail(request, movie_id):
    movie = Movie.objects.get(movie_id=movie_id)
    return {"movie": serialize_movie(movie)}


@api_view
def similar_movies(request, movie_id):
    n = int_param(request, "n", default=10, maximum=MAX_RESULTS)
    seed = Movie.objects.get(movie_id=movie_id)
    results = get_engine().similar_movies(movie_id, n=n)
    return {"seed_movie": serialize_movie(seed), "count": len(results),
            "results": [serialize_movie(m) for m in results]}


@api_view
def genres(request):
    names = set()
    for value in Movie.objects.values_list("genres", flat=True).distinct():
        names.update(g for g in value.split("|") if g and g != "(no genres listed)")
    return {"results": sorted(names)}


# ----- likes + live defense ------------------------------------------------


@api_view(methods=("POST",))
def like_movie(request):
    """Like (1), dislike (-1) or remove (0) a user's reaction to a movie.

    Every click is logged, then the live shilling defense judges the user's recent
    activity. A warning still applies the click; a restriction rejects it (HTTP 429).
    """
    body = json_body(request)
    user_id = body_int(body, "user_id")
    movie_id = body_int(body, "movie_id")
    value = body_int(body, "value", minimum=-1, maximum=1)
    movie = Movie.objects.get(movie_id=movie_id)

    with transaction.atomic():
        LikeEvent.objects.create(user_id=user_id, movie=movie, action=LIKE_ACTIONS[value])
        verdict = assess_live_activity(user_id, movie_id)
        if verdict.allowed:
            if value == 0:
                Like.objects.filter(user_id=user_id, movie=movie).delete()
            else:
                Like.objects.update_or_create(user_id=user_id, movie=movie, defaults={"value": value})

    current = Like.objects.filter(user_id=user_id, movie=movie).values_list("value", flat=True).first()
    retrain_scheduled = False
    if verdict.allowed and verdict.status != UserTrust.RESTRICTED:
        retrain_scheduled = note_training_data_changed()

    engine = get_engine()
    payload = {
        "user_id": user_id,
        "movie": serialize_movie(movie),
        "value": current or 0,
        "applied": verdict.allowed,
        "trust": verdict.to_dict(),
        "taste": engine.taste_summary(user_id),
        "model": {"retrain_scheduled": retrain_scheduled, **engine_status()},
    }
    return JsonResponse(payload, status=200 if verdict.allowed else 429)


@api_view
def user_profile(request, user_id):
    engine = get_engine()
    likes = Like.objects.filter(user_id=user_id).select_related("movie").order_by("-updated_at")
    return {
        "user_id": user_id,
        "trust": describe_trust(user_id).to_dict(),
        "taste": engine.taste_summary(user_id),
        "likes": [
            {"movie": serialize_movie(like.movie), "value": like.value,
             "updated_at": like.updated_at.isoformat()}
            for like in likes
        ],
        "model": engine_status(),
    }


@api_view
def defense_report(request):
    engine = get_engine()
    return {
        "batch": engine.defense_report.to_dict(),
        "live": live_defense_overview(),
        "model": engine_status(),
    }
