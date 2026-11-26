from __future__ import annotations

from pathlib import Path

import pytest

from arc_cd.data_prep.dataset_ids import infer_concept_arc_dataset_id


def test_infer_concept_arc_dataset_id_keeps_numeric_variants() -> None:
    """Normalize ConceptARC variant path names into output dataset IDs."""
    assert infer_concept_arc_dataset_id("/data/ConceptARC") == "concept_arc"
    assert (
        infer_concept_arc_dataset_id("/work/conceptarc-100-eval/data/ConceptARC/corpus")
        == "concept_arc"
    )
    assert infer_concept_arc_dataset_id("/data/ConceptARC-40") == "concept_arc_40"
    assert infer_concept_arc_dataset_id("/data/concept_arc_30") == "concept_arc_30"
    assert (
        infer_concept_arc_dataset_id(
            Path("output/perturbations/concept_arc_40/concept_arc_40_seed7.jsonl")
        )
        == "concept_arc_40"
    )


def test_infer_concept_arc_dataset_id_rejects_unknown_paths() -> None:
    """Require an explicit ConceptARC marker before deriving an ID."""
    with pytest.raises(ValueError, match="ConceptARC dataset ID"):
        infer_concept_arc_dataset_id("/data/arc-agi-1/training")


@pytest.mark.parametrize("name", ["ConceptARC-100-8", "concept_arc_100_8"])
def test_support_version_dataset_ids(name: str) -> None:
    """Keep both numeric components when naming support-version artifacts."""
    assert infer_concept_arc_dataset_id(f"/data/{name}") == "concept_arc_100_8"
