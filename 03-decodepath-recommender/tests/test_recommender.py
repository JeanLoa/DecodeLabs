"""Behavior and architecture checks for DecodePath."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.catalog import CatalogValidationError, load_catalog
from src.models import UserProfile
from src.recommender import InsufficientProfileError, TechStackRecommender
from src.storage import RecommendationStore

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "raw" / "raw_skills.csv"
TEST_DATA_PATH = ROOT / ".test-data"
TEST_DATA_PATH.mkdir(exist_ok=True)


class CatalogTests(unittest.TestCase):
    def test_catalog_contains_complete_recommendable_items(self) -> None:
        roles = load_catalog(CATALOG_PATH)

        self.assertGreaterEqual(len(roles), 15)
        self.assertEqual(len({role.role_id for role in roles}), len(roles))
        self.assertTrue(all(len(role.skills) >= 3 for role in roles))
        self.assertTrue(all(role.learning_path for role in roles))

    def test_catalog_rejects_items_without_enough_metadata(self) -> None:
        content = (
            "role_id,role_name,category,summary,skills,tools,career_keywords,"
            "learning_path,popularity\n"
            "bad,Bad Role,Test,Incomplete,Python|SQL,Tool,keyword,step,50\n"
        )
        path = TEST_DATA_PATH / "bad-catalog.csv"
        path.write_text(content, encoding="utf-8")
        try:
            with self.assertRaises(CatalogValidationError):
                load_catalog(path)
        finally:
            path.unlink(missing_ok=True)


class RecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = TechStackRecommender(load_catalog(CATALOG_PATH))

    def test_profile_requires_three_distinct_skills(self) -> None:
        profile = UserProfile(skills=("Python", "Python", "SQL"))

        with self.assertRaises(InsufficientProfileError):
            self.engine.recommend(profile)

    def test_free_text_extracts_aliases_into_canonical_skills(self) -> None:
        extracted = self.engine.extract_skills(
            "I build with Py, Amazon Web Services, K8s and REST APIs."
        )

        self.assertIn("Python", extracted)
        self.assertIn("AWS", extracted)
        self.assertIn("Kubernetes", extracted)
        self.assertIn("REST APIs", extracted)

    def test_data_profile_returns_relevant_top_three(self) -> None:
        results = self.engine.recommend(
            UserProfile(
                skills=("Python", "SQL", "Statistics", "Machine Learning"),
                goal="Move into an AI role",
            )
        )

        names = {item.role.name for item in results}
        self.assertEqual(len(results), 3)
        self.assertIn("Data Scientist", names)
        self.assertTrue(all(0.0 <= item.score <= 1.0 for item in results))

    def test_frontend_profile_prioritizes_frontend_developer(self) -> None:
        results = self.engine.recommend(
            UserProfile(
                skills=("JavaScript", "TypeScript", "React", "CSS"),
                goal="Build accessible visual products",
            )
        )

        self.assertEqual(results[0].role.name, "Frontend Developer")
        self.assertEqual(
            set(results[0].matched_skills),
            {"JavaScript", "TypeScript", "React", "CSS"},
        )

    def test_ranking_is_deterministic_and_descending(self) -> None:
        profile = UserProfile(
            skills=("AWS", "Cloud Computing", "Terraform", "Kubernetes")
        )
        first = self.engine.recommend(profile)
        second = self.engine.recommend(profile)

        self.assertEqual(
            [item.role.role_id for item in first],
            [item.role.role_id for item in second],
        )
        self.assertEqual(
            [item.score for item in first],
            sorted((item.score for item in first), reverse=True),
        )

    def test_cold_start_fallback_uses_popularity(self) -> None:
        trending = self.engine.trending()

        self.assertEqual(len(trending), 3)
        self.assertEqual(
            [role.popularity for role in trending],
            sorted((role.popularity for role in trending), reverse=True),
        )


class StorageTests(unittest.TestCase):
    def test_sessions_round_trip_as_json_serializable_data(self) -> None:
        engine = TechStackRecommender(load_catalog(CATALOG_PATH))
        profile = UserProfile(skills=("Python", "SQL", "Statistics"))
        recommendations = engine.recommend(profile)

        database_path = TEST_DATA_PATH / "history.db"
        database_path.unlink(missing_ok=True)
        try:
            store = RecommendationStore(database_path)
            session_id = store.save(profile, recommendations)
            saved = store.get(session_id)

            self.assertIsNotNone(saved)
            assert saved is not None
            self.assertEqual(saved.profile["skills"], ["Python", "SQL", "Statistics"])
            self.assertEqual(len(saved.recommendations), 3)
            json.dumps(saved.recommendations)
            self.assertEqual(store.list_recent()[0].id, session_id)
        finally:
            database_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
