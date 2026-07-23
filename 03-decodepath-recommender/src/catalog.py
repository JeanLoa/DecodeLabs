"""Load and validate the local career catalog."""

from __future__ import annotations

import csv
from pathlib import Path

from .models import CareerRole

REQUIRED_COLUMNS = {
    "role_id",
    "role_name",
    "category",
    "summary",
    "skills",
    "tools",
    "career_keywords",
    "learning_path",
    "popularity",
}


class CatalogValidationError(ValueError):
    """Raised when a catalog row cannot participate in recommendations."""


def _split(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split("|") if item.strip())


def load_catalog(path: str | Path) -> tuple[CareerRole, ...]:
    """Return a validated, immutable catalog from a CSV file."""

    catalog_path = Path(path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Career catalog not found: {catalog_path}")

    with catalog_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise CatalogValidationError(
                f"Catalog is missing columns: {', '.join(sorted(missing))}"
            )

        roles: list[CareerRole] = []
        seen_ids: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            role_id = row["role_id"].strip()
            skills = _split(row["skills"])
            tools = _split(row["tools"])
            keywords = _split(row["career_keywords"])
            learning_path = _split(row["learning_path"])

            if not role_id or role_id in seen_ids:
                raise CatalogValidationError(
                    f"Line {line_number} has an empty or duplicate role_id."
                )
            if len(skills) < 3:
                raise CatalogValidationError(
                    f"Role {role_id!r} needs at least three skill tags."
                )
            if not tools or not keywords or not learning_path:
                raise CatalogValidationError(
                    f"Role {role_id!r} has incomplete recommendation metadata."
                )

            try:
                popularity = int(row["popularity"])
            except ValueError as error:
                raise CatalogValidationError(
                    f"Role {role_id!r} has an invalid popularity value."
                ) from error
            if not 0 <= popularity <= 100:
                raise CatalogValidationError(
                    f"Role {role_id!r} popularity must be between 0 and 100."
                )

            roles.append(
                CareerRole(
                    role_id=role_id,
                    name=row["role_name"].strip(),
                    category=row["category"].strip(),
                    summary=row["summary"].strip(),
                    skills=skills,
                    tools=tools,
                    keywords=keywords,
                    learning_path=learning_path,
                    popularity=popularity,
                )
            )
            seen_ids.add(role_id)

    if len(roles) < 3:
        raise CatalogValidationError("The catalog needs at least three roles.")
    return tuple(roles)
