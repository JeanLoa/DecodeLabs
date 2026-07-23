"""Run DecodePath from the terminal without starting Streamlit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import TechStackRecommender, UserProfile, load_catalog  # noqa: E402
from src.recommender import InsufficientProfileError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recommend three technology career paths from explicit skills."
    )
    parser.add_argument(
        "skills",
        nargs="+",
        help='At least three skills, for example: Python SQL "Machine Learning"',
    )
    parser.add_argument("--goal", default="", help="Optional career goal.")
    parser.add_argument("--interests", default="", help="Optional interests.")
    parser.add_argument(
        "--experience",
        default="Exploring",
        choices=["Exploring", "Beginner", "Intermediate", "Advanced"],
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    recommender = TechStackRecommender(
        load_catalog(ROOT / "data" / "raw" / "raw_skills.csv")
    )
    profile = UserProfile(
        skills=tuple(args.skills),
        goal=args.goal,
        interests=args.interests,
        experience=args.experience,
    )
    try:
        recommendations = recommender.recommend(profile)
    except InsufficientProfileError as error:
        print(f"Profile error: {error}", file=sys.stderr)
        return 2

    output = {
        "profile": profile.to_dict(),
        "method": "TF-IDF + cosine similarity",
        "recommendations": [item.to_dict() for item in recommendations],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
