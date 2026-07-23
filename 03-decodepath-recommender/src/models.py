"""Domain models used by the recommendation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CareerRole:
    """A recommendable technology career and its content metadata."""

    role_id: str
    name: str
    category: str
    summary: str
    skills: tuple[str, ...]
    tools: tuple[str, ...]
    keywords: tuple[str, ...]
    learning_path: tuple[str, ...]
    popularity: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UserProfile:
    """Explicit preferences gathered from the user."""

    skills: tuple[str, ...]
    goal: str = ""
    interests: str = ""
    experience: str = "Exploring"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Recommendation:
    """A scored role plus evidence that explains the match."""

    rank: int
    role: CareerRole
    score: float
    matched_skills: tuple[str, ...]
    skill_gaps: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": round(self.score, 6),
            "matched_skills": list(self.matched_skills),
            "skill_gaps": list(self.skill_gaps),
            "rationale": self.rationale,
            "role": self.role.to_dict(),
        }
