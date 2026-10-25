"""
JSON REST API for the movie recommender.

    GET /api/health/
    GET /api/recommendations/?user_id=1&movie_id=10[&n=5]
    GET /api/movies/popular/[?n=20&genre=Comedy]
    GET /api/movies/search/?q=toy[&n=10]
    GET /api/movies/<movie_id>/
    GET /api/movies/<movie_id>/similar/[?n=10]
    GET /api/genres/
"""
import logging
from functools import wraps

from django.db.models import Count
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET

from .ml_engine import MovieNotFound, get_engine
from .models import Movie, Rating

logger = logging.getLogger(__name__)

MAX_RESULTS = 50


class BadRequest(ValueError):
    pass


def error(message, status):
    return JsonResponse({"error": message}, status=status)


def api_view(view):
    """GET-only JSON view with uniform error responses."""
    # Read-only views: exempt from CSRF so non-GET requests get a clean 405, not a 403.
    @csrf_exempt
    @require_GET
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        try:
            return JsonResponse(view(request, *args, **kwargs))
        except BadRequest as exc:
            return error(str(exc), 400)
        except (MovieNotFound, Movie.DoesNotExist):
            return error("Movie not found.", 404)
        except RuntimeError as exc:
            logger.exception("Recommendation engine unavailable")
            return error(str(exc), 503)
    return wrapper


def int_param(request, name, default=None, minimum=1, maximum=None):
    raw = request.GET.get(name)
    if raw in (None, ""):
        if default is None:
            raise BadRequest(f"Missing required query parameter '{name}'.")
        return default
    try:
        value = int(raw)
    except ValueError:
        raise BadRequest(f"'{name}' must be an integer.") from None
    if value < minimum:
        raise BadRequest(f"'{name}' must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise BadRequest(f"'{name}' must be <= {maximum}.")
    return value


def serialize_movie(movie):
    data = {
        "movie_id": movie.movie_id,
        "title": movie.title,
        "year": movie.year,
        "genres": movie.genre_list,
    }
    for attr in ("score", "similarity", "predicted_rating", "strategy"):
        if hasattr(movie, attr):
            data[attr] = getattr(movie, attr)
    return data


@api_view
def health(request):
    return {
        "status": "ok",
        "movies": Movie.objects.count(),
        "ratings": Rating.objects.count(),
    }


@api_view
def recommendations(request):
    user_id = int_param(request, "user_id")
    movie_id = int_param(request, "movie_id")
    n = int_param(request, "n", default=5, maximum=MAX_RESULTS)

    seed = Movie.objects.get(movie_id=movie_id)
    results = get_engine().recommend(user_id, movie_id, n=n)
    return {
        "user_id": user_id,
        "seed_movie": serialize_movie(seed),
        "strategy": results[0].strategy if results else None,
        "count": len(results),
        "results": [serialize_movie(m) for m in results],
    }


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
