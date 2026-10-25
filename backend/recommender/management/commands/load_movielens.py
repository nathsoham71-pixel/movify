"""
Ingest the MovieLens ml-latest-small dataset into the database.

Usage:
    python manage.py load_movielens
    python manage.py load_movielens --clear
    python manage.py load_movielens --data-dir path/to/ml-latest-small --batch-size 10000
"""
import re
import time
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from recommender.models import Movie, Rating

YEAR_PATTERN = re.compile(r"\((\d{4})\)\s*$")


def split_title_and_year(raw_title):
    """'Toy Story (1995)' -> ('Toy Story', 1995). Titles without a year return None."""
    raw_title = str(raw_title).strip()
    match = YEAR_PATTERN.search(raw_title)
    if not match:
        return raw_title, None
    return raw_title[: match.start()].strip(), int(match.group(1))


class Command(BaseCommand):
    help = "Load MovieLens movies.csv and ratings.csv into the database using bulk_create."

    def add_arguments(self, parser):
        parser.add_argument(
            "--data-dir",
            type=str,
            default=str(settings.MOVIELENS_DATA_DIR),
            help="Directory containing movies.csv and ratings.csv.",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=5000,
            help="Number of rows inserted per bulk_create batch.",
        )
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Delete all existing movies and ratings before loading.",
        )

    def handle(self, *args, **options):
        data_dir = Path(options["data_dir"])
        batch_size = options["batch_size"]
        movies_csv = data_dir / "movies.csv"
        ratings_csv = data_dir / "ratings.csv"

        for csv_path in (movies_csv, ratings_csv):
            if not csv_path.exists():
                raise CommandError(f"File not found: {csv_path}")

        if batch_size <= 0:
            raise CommandError("--batch-size must be a positive integer.")

        start = time.perf_counter()

        with transaction.atomic():
            if options["clear"]:
                self.stdout.write("Clearing existing ratings and movies...")
                Rating.objects.all().delete()
                Movie.objects.all().delete()

            valid_movie_ids = self.load_movies(movies_csv, batch_size)
            self.load_ratings(ratings_csv, batch_size, valid_movie_ids)

        elapsed = time.perf_counter() - start
        self.stdout.write(self.style.SUCCESS(
            f"Done in {elapsed:.1f}s. Database now has "
            f"{Movie.objects.count():,} movies and {Rating.objects.count():,} ratings."
        ))

    def load_movies(self, csv_path, batch_size):
        self.stdout.write(f"Reading {csv_path} ...")
        df = pd.read_csv(csv_path, dtype={"movieId": "int64", "title": str, "genres": str})
        df = df.dropna(subset=["movieId", "title"]).drop_duplicates(subset="movieId")
        df["genres"] = df["genres"].fillna("")

        existing_ids = set(Movie.objects.values_list("movie_id", flat=True))
        new_movies = []
        for row in df.itertuples(index=False):
            if row.movieId in existing_ids:
                continue
            title, year = split_title_and_year(row.title)
            new_movies.append(Movie(
                movie_id=int(row.movieId),
                title=title[:255],
                year=year,
                genres=row.genres[:255],
            ))

        Movie.objects.bulk_create(new_movies, batch_size=batch_size)
        self.stdout.write(self.style.SUCCESS(
            f"  Inserted {len(new_movies):,} movies "
            f"({len(existing_ids):,} already present)."
        ))
        return existing_ids | {m.movie_id for m in new_movies}

    def load_ratings(self, csv_path, batch_size, valid_movie_ids):
        self.stdout.write(f"Reading {csv_path} ...")
        df = pd.read_csv(
            csv_path,
            dtype={"userId": "int64", "movieId": "int64", "rating": "float64", "timestamp": "int64"},
        )
        total_rows = len(df)

        df = df[df["movieId"].isin(valid_movie_ids)]
        df = df[df["rating"].between(0.5, 5.0)]
        df = df.drop_duplicates(subset=["userId", "movieId"], keep="last")
        skipped = total_rows - len(df)

        inserted = 0
        batch = []
        for row in df.itertuples(index=False):
            batch.append(Rating(
                user_id=int(row.userId),
                movie_id=int(row.movieId),
                rating=float(row.rating),
                timestamp=int(row.timestamp),
            ))
            if len(batch) >= batch_size:
                inserted += self._flush_ratings(batch, batch_size)
                batch = []
        if batch:
            inserted += self._flush_ratings(batch, batch_size)

        self.stdout.write(self.style.SUCCESS(
            f"  Processed {inserted:,} ratings ({skipped:,} invalid/duplicate rows skipped)."
        ))

    def _flush_ratings(self, batch, batch_size):
        # ignore_conflicts makes re-running the command safe without --clear.
        Rating.objects.bulk_create(batch, batch_size=batch_size, ignore_conflicts=True)
        return len(batch)
