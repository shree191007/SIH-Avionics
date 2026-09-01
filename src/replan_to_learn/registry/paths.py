"""
SIH26054 Replan to Learn: Artifact registry path resolution (M3, F10).
Implements the 7-directory hierarchy under artifacts/ from
01_foundation_data_contracts.md Sec 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

ARTIFACT_CATEGORIES: Tuple[str, ...] = (
    "raw",
    "processed",
    "calibration",
    "models",
    "fingerprints",
    "replay",
    "evaluation",
)


class ArtifactPaths:
    """
    Resolves paths under a root `artifacts/` directory.

    ```
    artifacts/
      raw/
      processed/
      calibration/
      models/
      fingerprints/
      replay/
      evaluation/
    ```
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def ensure_hierarchy(self) -> None:
        """Create the 7-category directory hierarchy if it doesn't already exist."""
        self.root.mkdir(parents=True, exist_ok=True)
        for category in ARTIFACT_CATEGORIES:
            self.category_dir(category).mkdir(parents=True, exist_ok=True)

    def category_dir(self, category: str) -> Path:
        self._validate_category(category)
        return self.root / category

    def artifact_path(self, category: str, filename: str) -> Path:
        return self.category_dir(category) / filename

    def manifest_sidecar_path(self, artifact_path: Path) -> Path:
        return artifact_path.with_name(f"{artifact_path.name}.manifest.json")

    def lineage_index_path(self) -> Path:
        """Path to the persisted lineage DAG index (reproducibility/lineage.py)."""
        return self.root / "replay" / "lineage_index.json"

    @staticmethod
    def _validate_category(category: str) -> None:
        if category not in ARTIFACT_CATEGORIES:
            raise ValueError(
                f"Unknown artifact category '{category}'; must be one of {ARTIFACT_CATEGORIES}"
            )
