"""Cosine-based task embeddings and clusters independent of the 2D projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

from arc_cd.repr_analysis.geometry import finite_mean


@dataclass(frozen=True)
class EmbeddingOptions:
    """Numerical settings shared by every layer and support position.

    Attributes:
        methods (tuple[str, ...]): Ordered reducers, umap and/or pca.
        seed (int): Random seed for UMAP.
        n_neighbors (int): UMAP neighborhood size, capped at valid task count minus one.
        min_dist (float): UMAP minimum distance in the display.
        min_cluster_size (int): Minimum HDBSCAN group size.
        min_samples (int): HDBSCAN density-neighborhood size, including the point itself.
        cluster_selection_method (str): HDBSCAN eom or leaf cluster selection.
    """

    methods: tuple[str, ...] = ("umap", "pca")
    seed: int = 42
    n_neighbors: int = 15
    min_dist: float = 0.1
    min_cluster_size: int = 10
    min_samples: int = 5
    cluster_selection_method: str = "eom"

    def __post_init__(self) -> None:
        """Reject unsupported reducers and invalid algorithm parameters."""
        if (
            not self.methods
            or len(set(self.methods)) != len(self.methods)
            or not set(self.methods) <= {"umap", "pca"}
            or self.seed < 0
            or self.n_neighbors < 2
            or not 0 <= self.min_dist <= 1
            or self.min_cluster_size < 2
            or self.min_samples < 1
            or self.cluster_selection_method not in ("eom", "leaf")
        ):
            raise ValueError("Invalid dimensionality-reduction or clustering options.")


def embed_tasks(
    task_means: np.ndarray, options: EmbeddingOptions
) -> dict[str, np.ndarray]:
    """Project normalized task means and cluster their original cosine distances.

    Concept labels are deliberately not inputs. Zero-norm tasks retain their row
    but have missing coordinates and cluster label -2; HDBSCAN noise is -1.
    Degenerate or too-small populations remain unprojected/unclustered rather
    than fabricating coordinates or groups.

    Args:
        task_means (np.ndarray): Centered, unnormalized [tasks, hidden] means.
        options (EmbeddingOptions): Reducer and clustering settings.

    Returns:
        dict[str, np.ndarray]: Coordinates [methods, tasks, two], valid-vector
        mask, labels, membership strengths, PCA explained-variance ratios,
        and effective UMAP neighborhood size.
    """
    means = np.asarray(task_means, dtype=np.float64)
    if means.ndim != 2 or not means.shape[1] or not np.isfinite(means).all():
        raise ValueError("Expected finite [tasks, hidden] task means.")
    norms = np.linalg.norm(means, axis=1)
    valid = norms > 0
    indices = np.flatnonzero(valid)
    vectors = means[valid] / norms[valid, None]
    n_valid = len(indices)
    coordinates = np.full((len(options.methods), len(means), 2), np.nan)
    clusters = np.full(len(means), -2, dtype=np.int64)
    clusters[valid] = -1
    strengths = np.full(len(means), np.nan)
    strengths[valid] = 0
    explained = np.full(2, np.nan)
    effective_neighbors = min(options.n_neighbors, max(n_valid - 1, 0))
    if n_valid:
        distances = np.clip(1 - vectors @ vectors.T, 0, 2)
        np.fill_diagonal(distances, 0)
        if n_valid >= max(options.min_cluster_size, options.min_samples):
            clustering = HDBSCAN(
                metric="precomputed",
                min_cluster_size=options.min_cluster_size,
                min_samples=options.min_samples,
                cluster_selection_method=options.cluster_selection_method,
                n_jobs=1,
                copy=True,
            ).fit(distances)
            # Number groups by their first task in the saved identity order.
            labels = clustering.labels_
            groups = sorted(
                set(labels) - {-1}, key=lambda c: np.flatnonzero(labels == c)[0]
            )
            for number, label in enumerate(groups):
                clusters[indices[labels == label]] = number
            strengths[valid] = clustering.probabilities_
        has_spread = n_valid > 1 and np.any(vectors != vectors[0])
        if has_spread:
            for index, method in enumerate(options.methods):
                if method == "pca":
                    if min(vectors.shape) < 2:
                        continue
                    model = PCA(n_components=2, svd_solver="full")
                    coordinates[index, valid] = model.fit_transform(vectors)
                    explained = model.explained_variance_ratio_
                elif n_valid >= 3:
                    from umap import UMAP

                    coordinates[index, valid] = UMAP(
                        n_components=2,
                        metric="precomputed",
                        n_neighbors=effective_neighbors,
                        min_dist=options.min_dist,
                        random_state=options.seed,
                        n_jobs=1,
                        init="random",
                    ).fit_transform(distances)
    return {
        "coordinates": coordinates,
        "valid_vectors": valid,
        "cluster_labels": clusters,
        "cluster_strengths": strengths,
        "pca_explained_variance_ratio": explained,
        "effective_n_neighbors": np.asarray(effective_neighbors),
    }


def representative_versions(versions: np.ndarray, task_means: np.ndarray) -> np.ndarray:
    """Select each task's version closest in cosine to its unnormalized mean.

    Args:
        versions (np.ndarray): Centered [tasks, eight versions, hidden] vectors.
        task_means (np.ndarray): Unnormalized [tasks, hidden] means.

    Returns:
        np.ndarray: Version offsets, or -1 if every comparison is undefined.
    """
    denominator = (
        np.linalg.norm(versions, axis=2) * np.linalg.norm(task_means, axis=1)[:, None]
    )
    similarity = np.divide(
        np.einsum("tvh,th->tv", versions, task_means),
        denominator,
        out=np.full(denominator.shape, np.nan),
        where=denominator > 0,
    )
    indices = np.argmax(np.where(np.isfinite(similarity), similarity, -np.inf), axis=1)
    indices[~np.isfinite(similarity).any(axis=1)] = -1
    return indices


def within_task_dispersion(versions: np.ndarray) -> np.ndarray:
    """Average pairwise cosine distance across the support sets of each task.

    Args:
        versions (np.ndarray): Centered features with shape [task, version, hidden].

    Returns:
        np.ndarray: One mean per task, or NaN if all pairwise cosines are undefined.
    """
    gram = versions @ versions.swapaxes(1, 2)
    norms2 = np.maximum(np.diagonal(gram, axis1=1, axis2=2), 0.0)
    a, b = np.triu_indices(versions.shape[1], k=1)
    denominator = np.sqrt(norms2[:, a] * norms2[:, b])
    pair_cosine = np.divide(
        gram[:, a, b],
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0,
    )
    dispersion, _ = finite_mean(1 - np.clip(pair_cosine, -1, 1), axis=1)
    return dispersion
