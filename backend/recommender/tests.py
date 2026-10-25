"""
Run with:  python manage.py test recommender

Covers the like system, like-driven personalisation, and the live shilling defense.
"""
from datetime import timedelta

from django.test import Client, TestCase, override_settings
from django.utils import timezone

from . import ml_engine
from .bot_defense import LiveDefenseConfig, get_training_dataframe
from .models import Like, LikeEvent, Movie, Rating, UserTrust

HORROR = ["Horror|Thriller", "Horror", "Horror|Mystery", "Horror|Thriller", "Horror|Sci-Fi"] * 2
COMEDY = ["Comedy|Romance", "Comedy", "Comedy|Drama", "Comedy|Romance", "Comedy|Family"] * 2
ACTION = ["Action|Adventure", "Action|Sci-Fi", "Action|Thriller", "Action", "Action|Crime"] * 2


# Background retraining is disabled: the test database transaction would block the thread.
@override_settings(RECOMMENDER_RETRAIN_EVERY=0)
class LikeSystemTestCase(TestCase):
    client_class = Client

    def setUp(self):
        self.client = Client(HTTP_HOST="127.0.0.1")
        movies = []
        mid = 1
        for genres in HORROR + COMEDY + ACTION:
            movies.append(Movie(movie_id=mid, title=f"Movie {mid}", year=2000, genres=genres))
            mid += 1
        Movie.objects.bulk_create(movies)
        self.movie_ids = [m.movie_id for m in movies]
        self.horror_ids = self.movie_ids[: len(HORROR)]

        # A small but real rating history so SVD can train.
        ratings = []
        for user in range(1, 6):
            for movie in self.movie_ids:
                ratings.append(Rating(user_id=user, movie_id=movie,
                                      rating=((user * 3 + movie * 7) % 9) / 2 + 1.0,
                                      timestamp=1_600_000_000))
        Rating.objects.bulk_create(ratings)
        ml_engine.reset_engine()

    def post_like(self, user_id, movie_id, value=1):
        return self.client.post("/api/likes/", data={"user_id": user_id, "movie_id": movie_id,
                                                      "value": value}, content_type="application/json")

    def genres_of(self, results):
        return [g for movie in results for g in movie["genres"]]

    # ----- likes change recommendations -----------------------------------

    def test_likes_personalise_for_you_row(self):
        user = 999
        before = self.client.get(f"/api/recommendations/for-you/?user_id={user}&n=5").json()
        self.assertEqual(before["strategy"], "popular")

        liked = self.horror_ids[:3]
        for movie_id in liked:
            self.assertEqual(self.post_like(user, movie_id).status_code, 200)

        after = self.client.get(f"/api/recommendations/for-you/?user_id={user}&n=5").json()
        self.assertEqual(after["strategy"], "taste_cold_start")
        self.assertEqual(after["taste"]["n_likes"], 3)
        self.assertEqual(after["taste"]["top_genres"][0]["genre"], "Horror")
        # Every pick now matches the liked genre, and the picks differ from before.
        self.assertTrue(all("Horror" in m["genres"] for m in after["results"]))
        self.assertNotEqual([m["movie_id"] for m in before["results"]],
                            [m["movie_id"] for m in after["results"]])
        # Liked movies are never recommended back.
        self.assertTrue(set(m["movie_id"] for m in after["results"]).isdisjoint(liked))

    def test_dislikes_push_a_genre_away(self):
        user = 998
        for movie_id in self.horror_ids[:3]:
            self.post_like(user, movie_id, 1)
        for movie_id in self.movie_ids[len(HORROR): len(HORROR) + 3]:   # dislike comedies
            self.post_like(user, movie_id, -1)
        result = self.client.get(f"/api/recommendations/for-you/?user_id={user}&n=5").json()
        self.assertEqual(result["taste"]["n_dislikes"], 3)
        self.assertFalse(any("Comedy" in m["genres"] for m in result["results"]))

    def test_like_toggle_and_remove(self):
        user, movie = 999, self.movie_ids[0]
        self.assertEqual(self.post_like(user, movie, 1).json()["value"], 1)
        self.assertEqual(self.post_like(user, movie, -1).json()["value"], -1)
        self.assertEqual(self.post_like(user, movie, 0).json()["value"], 0)
        self.assertFalse(Like.objects.filter(user_id=user).exists())
        self.assertEqual(LikeEvent.objects.filter(user_id=user).count(), 3)

    def test_trusted_likes_enter_training_data(self):
        self.post_like(999, self.movie_ids[0], 1)
        self.post_like(999, self.movie_ids[1], -1)
        df, report = get_training_dataframe()
        rows = df[df.user_id == 999].set_index("movie_id")["rating"]
        self.assertEqual(rows[self.movie_ids[0]], 5.0)
        self.assertEqual(rows[self.movie_ids[1]], 1.0)
        self.assertEqual(report.likes_used, 2)

    # ----- live shilling defense ------------------------------------------

    def test_rapid_clicking_warns_then_restricts(self):
        cfg = LiveDefenseConfig()
        user = 7777
        statuses = []
        for movie_id in self.movie_ids[: cfg.restrict_actions_per_window]:   # 15 fast clicks
            response = self.post_like(user, movie_id)
            statuses.append((response.status_code, response.json()["trust"]["status"]))

        # First `warn_actions_per_window` clicks are fine, then warnings, then restriction.
        self.assertEqual(statuses[cfg.warn_actions_per_window - 1], (200, "trusted"))
        self.assertEqual(statuses[cfg.warn_actions_per_window], (200, "warned"))
        self.assertTrue(all(code == 200 for code, _ in statuses[:cfg.restrict_actions_per_window]))

        # The 16th click crosses the hard limit: rejected and restricted.
        response = self.post_like(user, self.movie_ids[0], -1)
        self.assertEqual(response.status_code, 429)
        body = response.json()
        self.assertFalse(body["applied"])
        self.assertEqual(body["trust"]["status"], "restricted")
        self.assertIsNotNone(body["trust"]["restricted_until"])
        self.assertEqual(Like.objects.get(user_id=user, movie_id=self.movie_ids[0]).value, 1)

        # Still blocked on the next click.
        self.assertEqual(self.post_like(user, self.movie_ids[1], -1).status_code, 429)

    def test_restricted_user_is_quarantined_from_training_and_personalisation(self):
        user = 7777
        self.post_like(user, self.movie_ids[0])
        UserTrust.objects.filter(user_id=user).update(
            status=UserTrust.RESTRICTED, restricted_until=timezone.now() + timedelta(minutes=10),
            last_reason="test")
        df, report = get_training_dataframe()
        self.assertFalse((df.user_id == user).any())
        self.assertEqual(report.likes_quarantined, 1)

        taste = ml_engine.get_engine().taste_summary(user)
        self.assertTrue(taste["quarantined"])
        self.assertFalse(taste["active"])

    def test_restriction_expires_into_probation(self):
        user = 7777
        UserTrust.objects.create(user_id=user, status=UserTrust.RESTRICTED, strikes=1,
                                 restricted_until=timezone.now() - timedelta(seconds=1))
        response = self.post_like(user, self.movie_ids[0])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["trust"]["status"], "warned")

    def test_toggle_spam_on_one_movie_warns(self):
        user, movie = 7777, self.movie_ids[0]
        for i in range(LiveDefenseConfig().same_movie_actions_per_window - 1):
            self.post_like(user, movie, 1 if i % 2 == 0 else -1)
        response = self.post_like(user, movie, 1)
        self.assertEqual(response.json()["trust"]["status"], "warned")
        self.assertIn("same movie toggled", response.json()["trust"]["message"])

    def test_coordinated_push_by_fresh_accounts_is_flagged(self):
        target = self.movie_ids[len(HORROR)]
        self.assertEqual(self.post_like(5001, target).json()["trust"]["status"], "trusted")
        self.assertEqual(self.post_like(5002, target).json()["trust"]["status"], "trusted")
        third = self.post_like(5003, target).json()["trust"]
        self.assertEqual(third["status"], "warned")
        self.assertIn("new accounts targeted this movie", third["message"])

        # An established user (has MovieLens ratings) liking the same movie is not punished.
        self.assertEqual(self.post_like(1, target).json()["trust"]["status"], "trusted")

    def test_profile_and_defense_report_endpoints(self):
        self.post_like(999, self.movie_ids[0])
        profile = self.client.get("/api/users/999/profile/").json()
        self.assertEqual(len(profile["likes"]), 1)
        self.assertEqual(profile["trust"]["status"], "trusted")

        report = self.client.get("/api/defense/report/").json()
        self.assertIn("batch", report)
        self.assertIn("live", report)
        self.assertEqual(report["batch"]["likes_used"], 1)

    def test_like_validation(self):
        self.assertEqual(self.post_like(999, self.movie_ids[0], 5).status_code, 400)
        self.assertEqual(self.post_like(999, 424242, 1).status_code, 404)
        self.assertEqual(self.client.get("/api/likes/").status_code, 405)
