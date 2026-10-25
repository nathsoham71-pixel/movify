"""
Adds poster_url / backdrop_url / overview to movie dicts using TMDB (cached in the DB).

Set an API key in your environment:   setx TMDB_API_KEY "your_key"     (free at themoviedb.org)
Without a key the app still works: the frontend shows title cards instead of posters.
"""
import logging
from concurrent.futures import ThreadPoolExecutor

import requests
from django.conf import settings

from .models import MovieArtwork

log = logging.getLogger(__name__)

TMDB_API = "https://api.themoviedb.org/3/movie/{tmdb_id}"
TMDB_IMG = "https://image.tmdb.org/t/p"
POSTER_SIZE = "w342"
BACKDROP_SIZE = "w1280"


def _fetch_tmdb(item):
    """Runs in a worker thread (HTTP only, no DB access)."""
    movie_id, tmdb_id = item
    try:
        r = requests.get(
            TMDB_API.format(tmdb_id=tmdb_id),
            params={"api_key": settings.TMDB_API_KEY},
            timeout=5,
        )
        if r.status_code == 404:
            return movie_id, tmdb_id, {}            # unknown to TMDB -> cache the miss
        r.raise_for_status()
        return movie_id, tmdb_id, r.json()
    except requests.RequestException as exc:
        log.warning("TMDB lookup failed for %s: %s", tmdb_id, exc)
        return None                                   # transient error -> don't cache


def attach_artwork(movies):
    """Mutates every dict in `movies` (needs keys: id, tmdb_id)."""
    ids = {m["id"] for m in movies}
    cached = {a.movie_id: a for a in MovieArtwork.objects.filter(pk__in=ids)}

    if settings.TMDB_API_KEY:
        todo = {m["id"]: m["tmdb_id"] for m in movies if m["id"] not in cached and m["tmdb_id"]}
        if todo:
            with ThreadPoolExecutor(max_workers=12) as pool:
                results = [r for r in pool.map(_fetch_tmdb, todo.items()) if r]
            fresh = [
                MovieArtwork(
                    movie_id=mid,
                    tmdb_id=tid,
                    poster_path=data.get("poster_path") or "",
                    backdrop_path=data.get("backdrop_path") or "",
                    overview=data.get("overview") or "",
                )
                for mid, tid, data in results
            ]
            MovieArtwork.objects.bulk_create(fresh, ignore_conflicts=True)
            cached.update({a.movie_id: a for a in fresh})

    for m in movies:
        art = cached.get(m["id"])
        if not art:
            continue
        if art.poster_path:
            m["poster_url"] = f"{TMDB_IMG}/{POSTER_SIZE}{art.poster_path}"
        if art.backdrop_path:
            m["backdrop_url"] = f"{TMDB_IMG}/{BACKDROP_SIZE}{art.backdrop_path}"
        m["overview"] = art.overview
