"""Selectable visual descriptors and label-independent task preprocessing."""

from __future__ import annotations

from typing import Literal

import numpy as np

from arc_cd.data_prep.data_format import Grid
from arc_cd.repr_analysis.visual_objects import visual_scene_objects

SupportAggregation = Literal["concat", "mean_std"]
GridScope = Literal["input", "output", "both"]
GRID_SCOPES = ("input", "output", "both")
FEATURE_GROUPS = ("geometry", "color", "spatial", "objects", "pixel_change")
DEFAULT_FEATURES = FEATURE_GROUPS[:-1]
GEOMETRY_NAMES = ("height", "width", "area", "aspect_ratio")
SPATIAL_NAMES = (
    "foreground_bbox_height",
    "foreground_bbox_width",
    "foreground_centroid_x",
    "foreground_centroid_y",
)
OBJECT_NAMES = (
    "object_count",
    "largest_object_fraction",
    "mean_object_bbox_height",
    "mean_object_bbox_width",
)
PIXEL_CHANGE_NAMES = (
    "changed_count",
    "background_to_color_fraction",
    "color_to_background_fraction",
    "recolored_fraction",
)
FEATURE_DEFINITIONS = {
    "geometry": "height, width, area, width/height, separately per selected grid",
    "color": "fractions of grid cells with each color 0 through 9, separately per grid",
    "spatial": {
        "foreground_bbox_height": "inclusive foreground bounding-box height / grid height; 0 when empty",
        "foreground_bbox_width": "inclusive foreground bounding-box width / grid width; 0 when empty",
        "foreground_centroid_x": "mean zero-based foreground column / (width - 1); 0.5 when empty or width is 1",
        "foreground_centroid_y": "mean zero-based foreground row / (height - 1); 0.5 when empty or height is 1",
    },
    "objects": {
        "object_count": "number of visual_scene_objects; 0 when empty",
        "largest_object_fraction": "largest visible object cell count / grid area; 0 when empty",
        "mean_object_bbox_height": "unweighted mean inclusive object bounding-box height / grid height; 0 when empty",
        "mean_object_bbox_width": "unweighted mean inclusive object bounding-box width / grid width; 0 when empty",
    },
    "pixel_change": {
        "changed_count": "number of cells with unequal input/output colors",
        "background_to_color_fraction": "number of 0-to-nonzero cells / grid area",
        "color_to_background_fraction": "number of nonzero-to-0 cells / grid area",
        "recolored_fraction": "number of unequal nonzero-to-nonzero cells / grid area",
    },
}


def canonical_feature_groups(
    features: tuple[str, ...], grid_scope: GridScope
) -> tuple[str, ...]:
    """Validate selected groups and return them in stable coordinate order.

    Args:
        features (tuple[str, ...]): Nonempty, unique feature-group selection.
        grid_scope (GridScope): Input, output, or both.

    Returns:
        tuple[str, ...]: Groups in FEATURE_GROUPS order.
    """
    if grid_scope not in GRID_SCOPES:
        raise ValueError(f"Unknown grid scope: {grid_scope}")
    if (
        not features
        or len(set(features)) != len(features)
        or not set(features) <= set(FEATURE_GROUPS)
    ):
        raise ValueError("Feature groups must be nonempty, unique, and known.")
    if "pixel_change" in features and grid_scope != "both":
        raise ValueError("pixel_change features require --grid_scope both.")
    return tuple(group for group in FEATURE_GROUPS if group in features)


def _grid_array(grid: Grid | None) -> np.ndarray:
    """Validate an integer ARC grid without coercing booleans or fractional cells.

    Args:
        grid (Grid | None): Required nonempty grid containing colors 0 through 9;
            None is rejected when the selected scope requires this grid.

    Returns:
        np.ndarray: Two-dimensional integer grid.
    """
    values = np.asarray(grid, dtype=object)
    if values.ndim != 2 or not all(values.shape):
        raise ValueError("Expected a nonempty rectangular grid.")
    if any(
        isinstance(cell, (bool, np.bool_))
        or not isinstance(cell, (int, np.integer))
        or not 0 <= cell <= 9
        for cell in values.flat
    ):
        raise ValueError("Grid cells must be integer ARC colors 0 through 9.")
    return values.astype(np.int64)


def pair_feature_layout(
    grid_scope: GridScope = "both",
    features: tuple[str, ...] = DEFAULT_FEATURES,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Describe coordinates available from the selected side of a support pair.

    Args:
        grid_scope (GridScope): Input only, output only, or both grids.
        features (tuple[str, ...]): Requested descriptor groups.

    Returns:
        tuple[tuple[str, ...], tuple[str, ...]]: Coordinate names and block labels.
    """
    groups = canonical_feature_groups(features, grid_scope)
    sides = ("input", "output") if grid_scope == "both" else (grid_scope,)
    names: list[str] = []
    blocks: list[str] = []
    coordinates: tuple[str, ...]
    descriptors: tuple[str, ...]
    for group in groups:
        if group == "pixel_change":
            coordinates = PIXEL_CHANGE_NAMES
        else:
            if group == "geometry":
                descriptors = GEOMETRY_NAMES
            elif group == "color":
                descriptors = tuple(f"color_{color}_fraction" for color in range(10))
            elif group == "spatial":
                descriptors = SPATIAL_NAMES
            else:
                descriptors = OBJECT_NAMES
            coordinates = tuple(
                f"{side}_{name}" for side in sides for name in descriptors
            )
        names.extend(coordinates)
        blocks.extend([group] * len(coordinates))
    return tuple(names), tuple(blocks)


PAIR_FEATURE_NAMES, PAIR_FEATURE_BLOCKS = pair_feature_layout(features=FEATURE_GROUPS)


def _spatial_features(grid: np.ndarray) -> np.ndarray:
    """Measure foreground extent and location on a validated grid.

    Args:
        grid (np.ndarray): Nonempty two-dimensional integer ARC grid.

    Returns:
        np.ndarray: Normalized bounding-box height/width and centroid X/Y.
    """
    rows, cols = np.nonzero(grid)
    if not len(rows):
        return np.array([0.0, 0.0, 0.5, 0.5])
    height, width = grid.shape
    return np.array(
        [
            (rows.max() - rows.min() + 1) / height,
            (cols.max() - cols.min() + 1) / width,
            cols.mean() / (width - 1) if width > 1 else 0.5,
            rows.mean() / (height - 1) if height > 1 else 0.5,
        ]
    )


def _object_features(grid: np.ndarray) -> np.ndarray:
    """Summarize visible objects using the local extraction helper.

    Args:
        grid (np.ndarray): Validated integer ARC grid.

    Returns:
        np.ndarray: Count, largest area fraction, and mean normalized box extents.
    """
    objects = visual_scene_objects(grid.tolist())
    if not objects:
        return np.zeros(4, dtype=np.float64)
    sizes = [len(obj.pixels) for obj in objects]
    extents = np.array([np.ptp(np.asarray(obj.pixels), axis=0) + 1 for obj in objects])
    return np.r_[
        len(objects), max(sizes) / grid.size, extents.mean(axis=0) / grid.shape
    ]


def basic_pair_features(
    input_grid: Grid | None,
    output_grid: Grid | None,
    grid_scope: GridScope = "both",
    features: tuple[str, ...] = DEFAULT_FEATURES,
) -> np.ndarray:
    """Extract descriptors using only the grid sides selected by grid_scope.

    Args:
        input_grid (Grid | None): Support input; ignored for output-only features.
        output_grid (Grid | None): Support output; ignored for input-only features.
        grid_scope (GridScope): Selected grids; both requires equal dimensions.
        features (tuple[str, ...]): Requested groups; color zero is background.

    Returns:
        np.ndarray: Float64 vector ordered by pair_feature_layout.
    """
    groups = canonical_feature_groups(features, grid_scope)
    grids = []
    if grid_scope in ("input", "both"):
        grids.append(_grid_array(input_grid))
    if grid_scope in ("output", "both"):
        grids.append(_grid_array(output_grid))
    if grid_scope == "both" and grids[0].shape != grids[1].shape:
        raise ValueError("Visual features require equal input/output dimensions.")
    area = grids[0].size
    descriptors: list[np.ndarray] = []
    if "geometry" in groups:
        descriptors.extend(
            np.asarray(
                [
                    grid.shape[0],
                    grid.shape[1],
                    grid.size,
                    grid.shape[1] / grid.shape[0],
                ],
                dtype=np.float64,
            )
            for grid in grids
        )
    if "color" in groups:
        descriptors.extend(
            np.bincount(grid.ravel(), minlength=10) / grid.size for grid in grids
        )
    if "spatial" in groups:
        descriptors.extend(_spatial_features(grid) for grid in grids)
    if "objects" in groups:
        descriptors.extend(_object_features(grid) for grid in grids)
    if "pixel_change" in groups:
        source, target = grids
        changed = source != target
        descriptors.append(
            np.asarray(
                [
                    np.count_nonzero(changed),
                    np.count_nonzero((source == 0) & (target != 0)) / area,
                    np.count_nonzero((source != 0) & (target == 0)) / area,
                    np.count_nonzero((source != 0) & (target != 0) & changed) / area,
                ],
                dtype=np.float64,
            )
        )
    return np.concatenate(descriptors)


def support_feature_layout(
    aggregation: SupportAggregation,
    grid_scope: GridScope = "both",
    features: tuple[str, ...] = DEFAULT_FEATURES,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Name every aggregated coordinate and its original descriptor block.

    Args:
        aggregation (SupportAggregation): Ordered concatenation or mean and SD.
        grid_scope (GridScope): Which support grids supply descriptors.
        features (tuple[str, ...]): Requested groups.

    Returns:
        tuple[tuple[str, ...], tuple[str, ...]]: Feature names and block labels.
    """
    names, blocks = pair_feature_layout(grid_scope, features)
    prefixes: tuple[str, ...]
    if aggregation == "concat":
        prefixes = ("support_1", "support_2", "support_3")
    elif aggregation == "mean_std":
        prefixes = ("mean", "std")
    else:
        raise ValueError(f"Unknown support aggregation: {aggregation}")
    return (
        tuple(f"{prefix}/{name}" for prefix in prefixes for name in names),
        blocks * len(prefixes),
    )


def aggregate_support_features(
    features: np.ndarray,
    aggregation: SupportAggregation,
    grid_scope: GridScope = "both",
    feature_groups: tuple[str, ...] = DEFAULT_FEATURES,
) -> np.ndarray:
    """Combine three pair descriptors without changing their support order.

    Args:
        features (np.ndarray): Finite [three supports, selected descriptors] array.
        aggregation (SupportAggregation): Concatenate supports or their mean/SD.
        grid_scope (GridScope): Which support grids supply descriptors.
        feature_groups (tuple[str, ...]): Requested groups defining the columns.

    Returns:
        np.ndarray: Float64 version vector; SD uses the population divisor.
    """
    values = np.asarray(features, dtype=np.float64)
    names, _ = pair_feature_layout(grid_scope, feature_groups)
    if values.shape != (3, len(names)) or not np.isfinite(values).all():
        raise ValueError("Expected finite features for exactly three support pairs.")
    support_feature_layout(aggregation, grid_scope, feature_groups)
    if aggregation == "concat":
        return values.reshape(-1).copy()
    deviations = values.std(axis=0, ddof=0)
    # Repeated fractions can otherwise acquire a tiny nonzero rounding residual.
    deviations[np.ptp(values, axis=0) == 0] = 0.0
    return np.concatenate((values.mean(axis=0), deviations))


def preprocess_task_features(
    features: np.ndarray,
    blocks: tuple[str, ...],
    exclude_groups: tuple[str, ...] = (),
    *,
    feature_names: tuple[str, ...] | None = None,
    exclude_pair_features: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    """Standardize parent vectors and balance their retained descriptor blocks.

    Args:
        features (np.ndarray): Finite raw [selected parents, features] means.
        blocks (tuple[str, ...]): Feature-group label per coordinate.
        exclude_groups (tuple[str, ...]): Descriptor groups omitted from distance
            calculations. At least one group must remain selected.
        feature_names (tuple[str, ...] | None): Aggregated coordinate names;
            required when excluding individual pair descriptors.
        exclude_pair_features (tuple[str, ...]): Pair descriptor names to omit
            from every support slot or both mean and standard deviation.

    Returns:
        dict[str, np.ndarray]: Prepared vectors, full-coordinate means/scales,
        included/retained masks, full-coordinate weights (zero when dropped),
        and the scalar zero_norm_threshold used before distance calculation.
        All-constant inputs yield an empty feature axis: cosine is undefined,
        while Euclidean distances are all zero.
        Vectors within the float64 centering-error bound become zero; the bound
        accounts for sample count, feature magnitude, scaling, and block weights.
    """
    values = np.asarray(features, dtype=np.float64)
    if (
        values.ndim != 2
        or not all(values.shape)
        or values.shape[1] != len(blocks)
        or not np.isfinite(values).all()
        or not set(blocks) <= set(FEATURE_GROUPS)
    ):
        raise ValueError("Expected finite parent vectors with aligned feature blocks.")
    if (
        len(set(exclude_groups)) != len(exclude_groups)
        or not set(exclude_groups) <= set(blocks)
        or not set(blocks) - set(exclude_groups)
    ):
        raise ValueError(
            "Excluded groups must be unique and leave a feature group selected."
        )
    block_labels = np.asarray(blocks)
    included = ~np.isin(block_labels, exclude_groups)
    if exclude_pair_features:
        if feature_names is None or len(feature_names) != len(blocks):
            raise ValueError("Aligned feature names are required for pair ablation.")
        pair_names = np.asarray([name.rsplit("/", 1)[-1] for name in feature_names])
        if len(set(exclude_pair_features)) != len(exclude_pair_features) or not set(
            exclude_pair_features
        ) <= set(pair_names[included]):
            raise ValueError("Excluded pair features must be unique and selected.")
        included &= ~np.isin(pair_names, exclude_pair_features)
        if not included.any():
            raise ValueError("At least one pair feature must remain selected.")
    means = values.mean(axis=0)
    variable = np.ptp(values, axis=0) != 0
    retained = variable & included
    scales = values.std(axis=0, ddof=0)
    scales[~variable] = 0.0
    weights: np.ndarray = np.zeros(len(blocks), dtype=np.float64)
    for block in FEATURE_GROUPS:
        selected = retained & (block_labels == block)
        if selected.any():
            weights[selected] = 1 / np.sqrt(np.count_nonzero(selected))
    vectors = (values[:, retained] - means[retained]) / scales[retained]
    vectors *= weights[retained]
    epsilon = float(np.finfo(np.float64).eps)
    accumulation_error = len(values) * epsilon / (1 - len(values) * epsilon)
    # Bound error in an ordinary sum followed by division, then express it in
    # the same standardized and block-weighted coordinates as the task vectors.
    mean_error = (accumulation_error + epsilon) * np.abs(values).mean(axis=0)
    norm_error = np.linalg.norm(
        mean_error[retained] / scales[retained] * weights[retained]
    )
    zero_threshold = max(16 * epsilon, float(norm_error))
    vectors[np.linalg.norm(vectors, axis=1) <= zero_threshold] = 0.0
    return {
        "vectors": vectors,
        "mean": means,
        "scale": scales,
        "included": included,
        "retained": retained,
        "block_weights": weights,
        "zero_norm_threshold": np.asarray(zero_threshold),
    }
