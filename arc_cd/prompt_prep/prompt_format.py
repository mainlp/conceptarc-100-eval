"""
Functions and modules to construct prompts for ARC-style data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from arc_cd.data_prep import Grid

Message = dict[str, str]


@dataclass(frozen=True)
class PromptTemplates:
    """Prompt templates for ARC-style inference.

    Attributes:
        system: System prompt template.
        user: User prompt template. May contain placeholders such as
            ``{examples}``, ``{query_number}``, ``{test_input}``,
            ``{concept}``, ``{sample_id}``, ``{task_id}``,
            ``{case_id}``, and ``{test_index}``.
    """

    system: str
    user: str

    @classmethod
    def from_dir(cls, prompt_dir: str | Path) -> PromptTemplates:
        """Load prompt templates from a directory.

        Args:
            prompt_dir: Directory containing ``system.txt`` and ``user.txt``.

        Returns:
            Loaded prompt templates.

        Raises:
            FileNotFoundError: If either template file is missing.
        """
        prompt_path = Path(prompt_dir).expanduser().resolve()
        return cls(
            system=(prompt_path / "system.txt").read_text(encoding="utf-8").strip(),
            user=(prompt_path / "user.txt").read_text(encoding="utf-8").strip(),
        )


@dataclass(frozen=True)
class PromptRecord:
    """One model-ready ARC-style prompt record.

    Attributes:
        case_id: Stable flattened test-case identifier.
        task_id: Stable task identifier.
        sample_id: Sample identifier. For ARC-AGI-1, this is the task ID.
        test_index: Test example index within the task.
        messages: Chat-style messages suitable for tokenizer chat templates,
            vLLM chat inference, or OpenAI-compatible APIs.
        target_grid: Ground-truth output grid, if available.
        target_text: Ground-truth output in prompt grid format, if available.
        concept: Concept group name, if available.
    """

    case_id: str
    task_id: str
    sample_id: str
    test_index: int
    source_path: str
    messages: list[Message]
    target_grid: Optional[Grid]
    target_text: Optional[str]
    concept: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the prompt record to a JSON-serializable dictionary.

        Returns:
            JSON-serializable prompt record.
        """
        return {
            "case_id": self.case_id,
            "task_id": self.task_id,
            "concept": self.concept,
            "sample_id": self.sample_id,
            "test_index": self.test_index,
            "source_path": self.source_path,
            "messages": list(self.messages),
            "target_grid": self.target_grid,
            "target_text": self.target_text,
            "metadata": dict(self.metadata),
        }
