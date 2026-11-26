"""Helpers for ARC-style dataset identifiers."""

from __future__ import annotations

import re
from pathlib import Path

_CONCEPT_ARC_PATTERN = re.compile(
    r"concept[_-]?arc(?:[-_](?P<variant>\d+(?:[-_]\d+)*))?",
    re.IGNORECASE,
)


def infer_concept_arc_dataset_id(path: str | Path) -> str:
    """Infer the ConceptARC output dataset ID from a path.

    Numeric ConceptARC variants keep the shared loader and prompt templates but
    receive distinct artifact IDs. For example, ``ConceptARC-40`` becomes
    ``concept_arc_40``, ``ConceptARC-100-10`` becomes ``concept_arc_100_10``,
    and plain ``ConceptARC`` becomes ``concept_arc``.

    Args:
        path (str | Path): Dataset or artifact path containing a ConceptARC
            path component or filename.

    Returns:
        str: Normalized dataset ID.

    Raises:
        ValueError: If no ConceptARC marker appears in ``path``.
    """
    location = Path(path)
    for part in reversed(location.parts):
        match = _CONCEPT_ARC_PATTERN.fullmatch(part)
        if match is not None:
            break
    else:
        # An artifact filename may include the dataset ID alongside a model and seed.
        match = _CONCEPT_ARC_PATTERN.search(location.name) if location.suffix else None

    if match is not None:
        variant = match.group("variant")
        return f"concept_arc_{variant.replace('-', '_')}" if variant else "concept_arc"

    raise ValueError(f"Could not infer ConceptARC dataset ID from path: {path}")
