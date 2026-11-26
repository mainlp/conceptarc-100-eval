"""Visual object grouping using ConceptARC-GEN's component and dilation rules."""

from __future__ import annotations

from dataclasses import dataclass

from arc_cd.data_prep.data_format import Grid

Pixel = tuple[int, int]


@dataclass(frozen=True)
class VisualObject:
    """A same-color visual object containing only its original visible cells.

    Attributes:
        color (int): Non-background ARC color.
        pixels (tuple[Pixel, ...]): Sorted row/column coordinates.
    """

    color: int
    pixels: tuple[Pixel, ...]


def _same_color_components(grid: Grid) -> dict[int, list[tuple[Pixel, ...]]]:
    """Find eight-connected foreground components, grouped by color.

    Args:
        grid (Grid): Nonempty rectangular ARC grid; zero is background.

    Returns:
        dict[int, list[tuple[Pixel, ...]]]: Components with sorted visible pixels.
    """
    height, width = len(grid), len(grid[0])
    seen: set[Pixel] = set()
    components: dict[int, list[tuple[Pixel, ...]]] = {}
    for row in range(height):
        for col in range(width):
            color = grid[row][col]
            if color == 0 or (row, col) in seen:
                continue
            queue = [(row, col)]
            seen.add((row, col))
            pixels: list[Pixel] = []
            while queue:
                pixel = queue.pop()
                pixels.append(pixel)
                for next_row in range(max(0, pixel[0] - 1), min(height, pixel[0] + 2)):
                    for next_col in range(
                        max(0, pixel[1] - 1), min(width, pixel[1] + 2)
                    ):
                        neighbor = (next_row, next_col)
                        if neighbor in seen or grid[next_row][next_col] != color:
                            continue
                        seen.add(neighbor)
                        queue.append(neighbor)
            components.setdefault(color, []).append(tuple(sorted(pixels)))
    return components


def _group_singletons(
    pixels: list[Pixel], height: int, width: int
) -> list[tuple[Pixel, ...]]:
    """Group singleton cells by four-connectivity of their clipped dilation masks.

    Args:
        pixels (list[Pixel]): Singleton components of one color.
        height (int): Grid height.
        width (int): Grid width.

    Returns:
        list[tuple[Pixel, ...]]: Groups containing only original singleton pixels.
    """
    original = set(pixels)
    dilated = {
        (next_row, next_col)
        for row, col in pixels
        for next_row in range(max(0, row - 1), min(height, row + 2))
        for next_col in range(max(0, col - 1), min(width, col + 2))
    }
    groups: list[tuple[Pixel, ...]] = []
    seen: set[Pixel] = set()
    for pixel in sorted(dilated):
        if pixel in seen:
            continue
        queue = [pixel]
        seen.add(pixel)
        mask_component: set[Pixel] = set()
        while queue:
            row, col = queue.pop()
            mask_component.add((row, col))
            for neighbor in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if neighbor in seen or neighbor not in dilated:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        group = tuple(sorted(original & mask_component))
        if group:
            groups.append(group)
    return groups


def visual_scene_objects(grid: Grid) -> list[VisualObject]:
    """Extract visual objects using the generator's grouping convention.

    Same-color eight-connected components with two or more cells remain separate.
    Only singleton components are grouped by four-connectivity of their combined
    clipped 3x3 dilation masks. Dilation never adds cells to the returned objects.

    Args:
        grid (Grid): Nonempty rectangular ARC grid; zero is background.

    Returns:
        list[VisualObject]: Objects sorted by bounding-box row, column, color,
        and visible cell count. An all-background grid yields no objects.
    """
    height, width = len(grid), len(grid[0])
    objects: list[VisualObject] = []
    for color, components in _same_color_components(grid).items():
        groups = [pixels for pixels in components if len(pixels) > 1]
        singletons = [pixels[0] for pixels in components if len(pixels) == 1]
        groups.extend(_group_singletons(singletons, height, width))
        objects.extend(VisualObject(color, pixels) for pixels in groups)
    return sorted(
        objects,
        key=lambda obj: (
            min(row for row, _ in obj.pixels),
            min(col for _, col in obj.pixels),
            obj.color,
            len(obj.pixels),
        ),
    )
