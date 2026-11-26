"""
Functions and modules to parse responses to ARC-style data.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Optional

from arc_cd.data_prep import Grid


@dataclass(frozen=True)
class ParsedGrid:
    """Parsed grid result.

    Attributes:
        grid: Parsed grid, or None if parsing failed.
        status: Parse status string.
        final_answer: Text after the reasoning/thinking section.
        diagnostic_grid: Diagnostic parsed grid, regardless of finish reason.
        diagnostic_status: Diagnostic parse status string.
    """

    grid: Optional[Grid]
    status: str
    final_answer: str
    diagnostic_grid: Optional[Grid]
    diagnostic_status: str

    @property
    def ok(self) -> bool:
        """Whether parsing succeeded."""
        return self.grid is not None

    @property
    def diagnostic_ok(self) -> bool:
        """Whether diagnostic parsing succeeded."""
        return self.diagnostic_grid is not None


def extract_final_answer(text: str, end_think_token: str = "</think>") -> str:
    """Extract final answer text after the reasoning section.

    If the end-thinking token is absent, the whole text is returned. This keeps
    the evaluator usable for non-reasoning models too.

    Args:
        text: Raw generated model text.
        end_think_token: Token/string marking the end of reasoning.

    Returns:
        Final-answer text.
    """
    index = text.rfind(end_think_token)
    if index == -1:
        return text.strip()

    return text[index + len(end_think_token) :].strip()


def _parse_row_text(row_text: str) -> Optional[list[int]]:
    """Parse one bracketed row body.

    Args:
        row_text: Text inside brackets.

    Returns:
        Parsed row, or None if the row is invalid.
    """
    tokens = [token for token in re.split(r"[\s,]+", row_text.strip()) if token]

    if not tokens:
        return None

    if not all(re.fullmatch(r"-?\d+", token) for token in tokens):
        return None

    return [int(token) for token in tokens]


def _is_row_separator(text: str) -> bool:
    """Return whether text can separate adjacent grid rows.

    Args:
        text: Text between two bracketed rows.

    Returns:
        True if the separator is compatible with one grid.
    """
    return bool(re.fullmatch(r"[\s,;]*", text))


def _find_row_grid_groups(text: str) -> list[Grid]:
    """Find contiguous groups of bracketed integer rows.

    Args:
        text: Text to search.

    Returns:
        Candidate row-grid groups in left-to-right order.
    """
    groups: list[Grid] = []
    current_group: Grid = []
    previous_end: Optional[int] = None

    row_pattern = re.compile(r"\[([^\[\]]+)\]")

    for match in row_pattern.finditer(text):
        row = _parse_row_text(match.group(1))

        if row is None:
            if current_group:
                groups.append(current_group)
                current_group = []
            previous_end = None
            continue

        separator = (
            text[previous_end : match.start()] if previous_end is not None else ""
        )

        is_contiguous = previous_end is None or _is_row_separator(separator)

        if not is_contiguous and current_group:
            groups.append(current_group)
            current_group = []

        current_group.append(row)
        previous_end = match.end()

    if current_group:
        groups.append(current_group)

    return groups


def _validate_grid(value: Any, *, require_arc_colors: bool) -> Optional[Grid]:
    """Validate and normalize a grid object."""
    if not isinstance(value, list) or not value:
        return None

    if not all(isinstance(row, list) and row for row in value):
        return None

    width = len(value[0])
    if any(len(row) != width for row in value):
        return None

    grid: Grid = []
    for row in value:
        parsed_row: list[int] = []
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int):
                return None
            if require_arc_colors and not 0 <= cell <= 9:
                return None
            parsed_row.append(cell)
        grid.append(parsed_row)

    return grid


def parse_last_row_grid_seq(
    text: str,
    *,
    require_arc_colors: bool,
) -> Optional[Grid]:
    """Parse the last valid sequence of bracketed grid rows.

    This accepts rows such as:

        [0 1 2] [3 4 5]

    and also comma-separated rows such as:

        [0, 1, 2] [3, 4, 5]

    Args:
        text: Text to parse.
        require_arc_colors: Whether grid values must be in [0, 9].

    Returns:
        Last valid parsed grid, or None if no valid grid was found.
    """
    groups = _find_row_grid_groups(text)

    for group in reversed(groups):
        grid = _validate_grid(group, require_arc_colors=require_arc_colors)
        if grid is not None:
            return grid

    return None


def _balanced_bracket_substrings(text: str) -> list[str]:
    """Extract balanced bracket substrings, longest first."""
    substrings: list[str] = []

    for start, char in enumerate(text):
        if char != "[":
            continue

        depth = 0
        for end in range(start, len(text)):
            if text[end] == "[":
                depth += 1
            elif text[end] == "]":
                depth -= 1

            if depth == 0:
                substrings.append(text[start : end + 1])
                break

    return sorted(substrings, key=len, reverse=True)


def _try_parse_last_literal_grid(
    text: str,
    *,
    require_arc_colors: bool,
) -> Optional[Grid]:
    """Try to parse the last valid Python/JSON-style nested-list grid.

    Args:
        text: Text to parse.
        require_arc_colors: Whether grid values must be in [0, 9].

    Returns:
        Last valid parsed grid, or None if no valid grid was found.
    """
    candidates = _balanced_bracket_substrings(text)

    valid_grids: list[Grid] = []

    for candidate in candidates:
        try:
            value = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue

        grid = _validate_grid(value, require_arc_colors=require_arc_colors)
        if grid is not None:
            valid_grids.append(grid)

    return valid_grids[-1] if valid_grids else None


def _parse_grid_from_final_answer(
    final_answer: str,
    *,
    require_arc_colors: bool,
) -> tuple[Optional[Grid], str]:
    """Parse an ARC grid from final-answer text.

    Args:
        final_answer: Text after the reasoning section.
        require_arc_colors: Whether grid values must be integers in [0, 9].

    Returns:
        Parsed grid and parser status.
    """
    row_grid = parse_last_row_grid_seq(
        final_answer,
        require_arc_colors=require_arc_colors,
    )
    if row_grid is not None:
        return row_grid, "parsed_last_row_grid"

    literal_grid = _try_parse_last_literal_grid(
        final_answer,
        require_arc_colors=require_arc_colors,
    )
    if literal_grid is not None:
        return literal_grid, "parsed_last_literal_grid"

    return None, "parse_failed"


def parse_grid_from_generation(
    record: dict[str, Any],
    *,
    end_think_token: str = "</think>",
    require_arc_colors: bool = True,
) -> ParsedGrid:
    """Parse a predicted ARC grid from a generation record.

    The primary parser looks for the last valid sequence of bracketed grid rows,
    e.g. ``[0 1 2] [3 4 5]``. This matches the ConceptARC prompt format and is
    robust to models adding reasoning, labels, or extra text before the answer.

    Args:
        record: Generation record containing ``generated_text`` and
            ``finish_reason``.
        end_think_token: End-thinking marker.
        require_arc_colors: Whether grid values must be integers in [0, 9].

    Returns:
        Parsed grid result. Non-stop generations are not officially parsed,
        but retain diagnostic parse fields.
    """
    generated_text = record.get("generated_text", "")

    final_answer = extract_final_answer(
        generated_text,
        end_think_token=end_think_token,
    )

    diagnostic_grid, diagnostic_status = _parse_grid_from_final_answer(
        final_answer,
        require_arc_colors=require_arc_colors,
    )

    if record.get("finish_reason") != "stop":
        return ParsedGrid(
            grid=None,
            status="unfinished_generation",
            final_answer=final_answer,
            diagnostic_grid=diagnostic_grid,
            diagnostic_status=diagnostic_status,
        )

    return ParsedGrid(
        grid=diagnostic_grid,
        status=diagnostic_status,
        final_answer=final_answer,
        diagnostic_grid=diagnostic_grid,
        diagnostic_status=diagnostic_status,
    )
