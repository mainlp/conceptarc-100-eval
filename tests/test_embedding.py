"""Independent checks for cosine embeddings and projection-independent clusters."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

from arc_cd.repr_analysis.embedding import EmbeddingOptions, embed_tasks


def test_cosine_geometry_and_projection_independent_clusters() -> None:
    """Match direct normalized PCA and HDBSCAN, preserving original row order."""
    rng = np.random.default_rng(42)
    means = np.r_[
        rng.normal([3, 0, 0], 0.05, (12, 3)),
        rng.normal([-3, 0, 0], 0.05, (12, 3)),
        [[0, 0, 0]],
    ]
    means *= np.linspace(0.5, 8, len(means))[:, None]
    options = EmbeddingOptions(methods=("pca",), min_cluster_size=5, min_samples=3)
    result = embed_tasks(means, options)
    vectors = means[:-1] / np.linalg.norm(means[:-1], axis=1)[:, None]
    pca = PCA(n_components=2, svd_solver="full").fit(vectors)
    np.testing.assert_allclose(
        result["coordinates"][0, :-1], pca.transform(vectors), atol=1e-12
    )
    np.testing.assert_allclose(
        result["pca_explained_variance_ratio"], pca.explained_variance_ratio_
    )
    assert np.isnan(result["coordinates"][0, -1]).all()
    assert result["cluster_labels"][-1] == -2
    assert np.isnan(result["cluster_strengths"][-1])
    distances = np.clip(1 - vectors @ vectors.T, 0, 2)
    np.fill_diagonal(distances, 0)
    reference = HDBSCAN(
        metric="precomputed", min_cluster_size=5, min_samples=3, copy=True
    ).fit(distances)
    np.testing.assert_array_equal(
        result["cluster_labels"][:-1, None] == result["cluster_labels"][None, :-1],
        reference.labels_[:, None] == reference.labels_[None, :],
    )
    assert set(result["cluster_labels"]) == {-2, 0, 1}
    np.testing.assert_allclose(
        result["cluster_strengths"][:-1], reference.probabilities_
    )
    scaled = embed_tasks(means * 2, options)
    np.testing.assert_allclose(result["coordinates"], scaled["coordinates"])


def test_umap_seed_and_method_independence() -> None:
    """UMAP is repeatable and does not change labels or the PCA reference."""
    means = np.random.default_rng(7).normal(size=(18, 6))
    options = EmbeddingOptions(n_neighbors=40, min_cluster_size=4, min_samples=2)
    first = embed_tasks(means, options)
    second = embed_tasks(means, options)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    pca = embed_tasks(
        means, EmbeddingOptions(methods=("pca",), min_cluster_size=4, min_samples=2)
    )
    np.testing.assert_array_equal(first["cluster_labels"], pca["cluster_labels"])
    np.testing.assert_allclose(first["coordinates"][1], pca["coordinates"][0])
    assert first["effective_n_neighbors"] == 17
    assert np.isfinite(first["coordinates"]).all()


@pytest.mark.parametrize("means", [np.zeros((4, 3)), np.ones((5, 3)), np.ones((1, 3))])
def test_degenerate_embeddings_remain_missing(means: np.ndarray) -> None:
    """Undefined or constant populations have no invented 2D structure.

    Args:
        means (np.ndarray): Degenerate input task means.
    """
    result = embed_tasks(means, EmbeddingOptions())
    assert np.isnan(result["coordinates"]).all()
    np.testing.assert_array_equal(
        result["cluster_labels"], -2 if not means.any() else -1
    )


def test_small_populations_and_invalid_inputs() -> None:
    """Keep PCA for two tasks, skip UMAP, and reject nonfinite inputs/options."""
    result = embed_tasks(np.eye(2), EmbeddingOptions())
    assert np.isnan(result["coordinates"][0]).all()
    assert np.isfinite(result["coordinates"][1]).all()
    assert (result["cluster_labels"] == -1).all()
    for means in (np.array([[np.nan, 0]]), np.zeros(3), np.zeros((3, 0))):
        with pytest.raises(ValueError):
            embed_tasks(means, EmbeddingOptions())
    with pytest.raises(ValueError):
        EmbeddingOptions(min_samples=0)
    with pytest.raises(ValueError):
        EmbeddingOptions(methods=("pca", "pca"))
