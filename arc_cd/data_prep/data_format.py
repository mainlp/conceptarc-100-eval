"""
Classes and modules to represent ARC-style data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

Grid: TypeAlias = list[list[int]]


@dataclass(frozen=True)
class GridPair:
    """A single ARC-style input-output pair.

    Attributes:
        input: Input grid.
        output: Output grid, if available. Test outputs may be absent in some
            evaluation settings.
    """

    input: Grid
    output: Grid | None = None
