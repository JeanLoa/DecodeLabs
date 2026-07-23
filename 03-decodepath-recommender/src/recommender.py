"""Content-based career recommendations with TF-IDF and cosine similarity."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import CareerRole, Recommendation, UserProfile

MINIMUM_SKILLS = 3

# Each alias resolves to the canonical label displayed in the interface.
SKILL_ALIASES = {
    "amazon web services": "AWS",
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "Google Cloud",
    "google cloud platform": "Google Cloud",
    "google cloud": "Google Cloud",
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "py": "Python",
    "python": "Python",
    "js": "JavaScript",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "reactjs": "React",
    "react": "React",
    "vuejs": "Vue",
    "vue": "Vue",
    "angular": "Angular",
    "nodejs": "Node.js",
    "node js": "Node.js",
    "node": "Node.js",
    "spring boot": "Spring Boot",
    "spring": "Spring Boot",
    "dotnet": ".NET",
    "net": ".NET",
    "c sharp": "C#",
    "csharp": "C#",
    "machine learning": "Machine Learning",
    "ml": "Machine Learning",
    "deep learning": "Deep Learning",
    "dl": "Deep Learning",
    "natural language processing": "NLP",
    "nlp": "NLP",
    "artificial intelligence": "Artificial Intelligence",
    "ai": "Artificial Intelligence",
    "data visualization": "Data Visualization",
    "data viz": "Data Visualization",
    "business intelligence": "Business Intelligence",
    "bi": "Business Intelligence",
    "cloud computing": "Cloud Computing",
    "cloud": "Cloud Computing",
    "ci cd": "CI/CD",
    "cicd": "CI/CD",
    "continuous integration": "CI/CD",
    "infrastructure as code": "Infrastructure as Code",
    "iac": "Infrastructure as Code",
    "quality assurance": "Quality Assurance",
    "qa": "Quality Assurance",
    "test automation": "Test Automation",
    "automated testing": "Test Automation",
    "cyber security": "Cybersecurity",
    "cybersecurity": "Cybersecurity",
    "infosec": "Cybersecurity",
    "sql": "SQL",
    "nosql": "NoSQL",
    "rest api": "REST APIs",
    "rest apis": "REST APIs",
    "apis": "REST APIs",
    "api": "REST APIs",
    "powerbi": "Power BI",
    "power bi": "Power BI",
}


class InsufficientProfileError(ValueError):
    """Raised when a profile has too little signal for personalization."""


def normalize_text(value: str) -> str:
    """Normalize accents and punctuation without changing semantic content."""

    decomposed = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^a-zA-Z0-9+#.]+", " ", ascii_text).lower().split())


def _deduplicate(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = normalize_text(value)
        if key and key not in seen:
            result.append(value)
            seen.add(key)
    return tuple(result)


class TechStackRecommender:
    """Rank local career metadata against an explicit user profile."""

    def __init__(self, roles: Iterable[CareerRole]) -> None:
        self.roles = tuple(roles)
        if len(self.roles) < 3:
            raise ValueError("At least three career roles are required.")

        canonical = {
            item
            for role in self.roles
            for item in (*role.skills, *role.tools)
        }
        self.skill_options = tuple(sorted(canonical, key=str.casefold))
        self._canonical_lookup = {
            normalize_text(option): option for option in self.skill_options
        }

        documents = [self._role_document(role) for role in self.roles]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
            token_pattern=r"(?u)\b[\w+#.][\w+#.-]*\b",
        )
        self.role_matrix = self.vectorizer.fit_transform(documents)

    @staticmethod
    def _role_document(role: CareerRole) -> str:
        # Repetition makes explicit skill tags stronger than descriptive prose.
        return " ".join(
            [
                *role.skills,
                *role.skills,
                *role.skills,
                *role.tools,
                role.category,
                *role.keywords,
                role.summary,
            ]
        )

    def canonicalize_skills(self, skills: Iterable[str]) -> tuple[str, ...]:
        """Resolve aliases and catalog labels to one stable vocabulary."""

        resolved: list[str] = []
        for skill in skills:
            normalized = normalize_text(skill)
            canonical = (
                SKILL_ALIASES.get(normalized)
                or self._canonical_lookup.get(normalized)
                or skill.strip()
            )
            if canonical:
                resolved.append(canonical)
        return _deduplicate(resolved)

    def extract_skills(self, text: str) -> tuple[str, ...]:
        """Extract catalog skills and common aliases from free-form text."""

        normalized = f" {normalize_text(text)} "
        candidates: list[tuple[int, str]] = []
        vocabulary = {
            **self._canonical_lookup,
            **SKILL_ALIASES,
        }
        # Longest phrases first prevents "cloud" from shadowing "google cloud".
        for phrase, canonical in sorted(
            vocabulary.items(), key=lambda item: len(item[0]), reverse=True
        ):
            pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
            match = re.search(pattern, normalized)
            if match:
                candidates.append((match.start(), canonical))
        return _deduplicate(
            canonical for _, canonical in sorted(candidates, key=lambda item: item[0])
        )

    def recommend(
        self,
        profile: UserProfile,
        *,
        top_n: int = 3,
    ) -> tuple[Recommendation, ...]:
        """Return the most similar roles, sorted by cosine score."""

        skills = self.canonicalize_skills(profile.skills)
        if len(skills) < MINIMUM_SKILLS:
            raise InsufficientProfileError(
                f"Choose at least {MINIMUM_SKILLS} distinct skills."
            )
        if not 1 <= top_n <= len(self.roles):
            raise ValueError("top_n must fit inside the career catalog.")

        user_document = " ".join(
            [
                *skills,
                *skills,
                *skills,
                profile.goal,
                profile.interests,
                profile.experience,
            ]
        )
        user_vector = self.vectorizer.transform([user_document])
        scores = cosine_similarity(user_vector, self.role_matrix).ravel()
        ordered_indices = sorted(
            range(len(self.roles)),
            key=lambda index: (-float(scores[index]), self.roles[index].name),
        )[:top_n]

        recommendations: list[Recommendation] = []
        normalized_skills = {normalize_text(skill): skill for skill in skills}
        for rank, index in enumerate(ordered_indices, start=1):
            role = self.roles[index]
            matched = tuple(
                skill
                for skill in role.skills
                if normalize_text(skill) in normalized_skills
            )
            gaps = tuple(skill for skill in role.skills if skill not in matched)[:4]
            match_copy = (
                ", ".join(matched)
                if matched
                else "your stated goal and adjacent interests"
            )
            recommendations.append(
                Recommendation(
                    rank=rank,
                    role=role,
                    score=max(0.0, min(1.0, float(scores[index]))),
                    matched_skills=matched,
                    skill_gaps=gaps,
                    rationale=(
                        f"{role.name} aligns with {match_copy}. "
                        "The score comes from TF-IDF-weighted content similarity."
                    ),
                )
            )
        return tuple(recommendations)

    def trending(self, *, top_n: int = 3) -> tuple[CareerRole, ...]:
        """Cold-start fallback when a user has not supplied enough preferences."""

        if not 1 <= top_n <= len(self.roles):
            raise ValueError("top_n must fit inside the career catalog.")
        return tuple(
            sorted(
                self.roles,
                key=lambda role: (-role.popularity, role.name),
            )[:top_n]
        )
