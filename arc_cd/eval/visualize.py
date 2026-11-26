"""
Grid drawing primitives for qualitative task comparisons.
"""

from __future__ import annotations

from typing import Any

from PIL import ImageDraw, ImageFont

from arc_cd.data_prep import Grid, GridPair

# Standard ARC color palette.
ARC_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),  # black
    1: (0, 116, 217),  # blue
    2: (255, 65, 54),  # red
    3: (46, 204, 64),  # green
    4: (255, 220, 0),  # yellow
    5: (170, 170, 170),  # gray
    6: (240, 18, 190),  # magenta
    7: (255, 133, 27),  # orange
    8: (127, 219, 255),  # cyan
    9: (135, 12, 37),  # maroon
}

TEXT = (20, 20, 20)
GRID_LINE = (80, 80, 80)
MISSING_BG = (245, 245, 245)

PAIR_GAP = 18
ARROW_W = 46
PAIR_TITLE_H = 30
GRID_LABEL_H = 28
EMPTY_GRID_W = 88
EMPTY_GRID_H = 44

Example = dict[str, Any]


def grid_shape(grid: Grid) -> tuple[int, int]:
    """Return the row and column count for an ARC grid.

    Args:
        grid: A 2D ARC grid.

    Returns:
        A tuple of ``(rows, columns)``.
    """
    return len(grid), max((len(row) for row in grid), default=0)


def is_empty_grid(grid: Grid) -> bool:
    """Return whether a grid has no drawable cells.

    Args:
        grid: A 2D ARC grid.

    Returns:
        Whether the grid has zero rows or zero columns.
    """
    rows, cols = grid_shape(grid)
    return rows == 0 or cols == 0


def grid_pixel_size(grid: Grid, cell_size: int) -> tuple[int, int]:
    """Return the rendered pixel size of a grid.

    Args:
        grid: A 2D ARC grid.
        cell_size: Pixel size for each ARC cell.

    Returns:
        A tuple of ``(width, height)`` in pixels.
    """
    if is_empty_grid(grid):
        return EMPTY_GRID_W, EMPTY_GRID_H

    rows, cols = grid_shape(grid)
    return cols * cell_size + 1, rows * cell_size + 1


def text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    """Measure text size.

    Args:
        draw: PIL drawing context.
        text: Text to measure.
        font: Font used for drawing.

    Returns:
        A tuple of ``(width, height)`` in pixels.
    """
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def load_font(font_size: int) -> ImageFont.ImageFont:
    """Load a clear bold font, falling back to PIL's default font.

    Args:
        font_size: Font size in pixels.

    Returns:
        A PIL font object.
    """
    for font_name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue

    return ImageFont.load_default()


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    width: int,
    font: ImageFont.ImageFont,
) -> None:
    """Draw horizontally centered text.

    Args:
        draw: PIL drawing context.
        text: Text to draw.
        x: Left coordinate of the centering region.
        y: Top coordinate.
        width: Width of the centering region.
        font: Font used for drawing.
    """
    text_w, _ = text_size(draw, text, font)
    draw.text((x + (width - text_w) // 2, y), text, fill=TEXT, font=font)


def grid_pair_to_example(pair: GridPair) -> Example:
    """Convert a grid pair into the dictionary format used by the renderer.

    Args:
        pair: Input-output grid pair.

    Returns:
        Renderer-compatible example dictionary.
    """
    example: Example = {"input": pair.input}
    if pair.output is not None:
        example["output"] = pair.output
    return example


def pair_size(example: Example, cell_size: int) -> tuple[int, int]:
    """Return the rendered size of one input-output pair.

    Args:
        example: A train or test example.
        cell_size: Pixel size for each ARC cell.

    Returns:
        A tuple of ``(width, height)`` in pixels.
    """
    input_w, input_h = grid_pixel_size(example["input"], cell_size)

    if "output" in example:
        output_w, output_h = grid_pixel_size(example["output"], cell_size)
    else:
        output_w, output_h = max(80, input_w), max(40, input_h)

    width = input_w + PAIR_GAP + ARROW_W + PAIR_GAP + output_w
    height = PAIR_TITLE_H + GRID_LABEL_H + max(input_h, output_h)
    return width, height


def draw_grid(
    draw: ImageDraw.ImageDraw,
    grid: Grid,
    x: int,
    y: int,
    cell_size: int,
) -> None:
    """Draw an ARC grid.

    Args:
        draw: PIL drawing context.
        grid: A 2D ARC grid.
        x: Left coordinate.
        y: Top coordinate.
        cell_size: Pixel size for each ARC cell.
    """
    if is_empty_grid(grid):
        font = load_font(font_size=18)
        draw.rectangle(
            [x, y, x + EMPTY_GRID_W, y + EMPTY_GRID_H],
            fill=MISSING_BG,
            outline=GRID_LINE,
        )
        draw_centered_text(
            draw, "empty", x, y + EMPTY_GRID_H // 2 - 7, EMPTY_GRID_W, font
        )
        return

    rows, cols = grid_shape(grid)

    for r in range(rows):
        for c in range(cols):
            value = grid[r][c] if c < len(grid[r]) else 0
            color = ARC_COLORS.get(value, (255, 255, 255))

            x0 = x + c * cell_size
            y0 = y + r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            draw.rectangle([x0, y0, x1, y1], fill=color, outline=GRID_LINE)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    """Draw a right-pointing arrow.

    Args:
        draw: PIL drawing context.
        x: Left coordinate of the arrow region.
        y: Top coordinate of the arrow region.
        width: Width of the arrow region.
        height: Height of the arrow region.
    """
    center_y = y + height // 2
    line_start_x = x + 4
    line_end_x = x + width - 12
    arrow_tip_x = x + width + 2
    arrow_half_h = width // 4
    line_width = max(3, width // 16)

    draw.line(
        [(line_start_x, center_y), (line_end_x, center_y)],
        fill=TEXT,
        width=line_width,
    )
    draw.polygon(
        [
            (arrow_tip_x, center_y),
            (line_end_x, center_y - arrow_half_h),
            (line_end_x, center_y + arrow_half_h),
        ],
        fill=TEXT,
    )


def draw_missing_output(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    font: ImageFont.ImageFont,
) -> None:
    """Draw a placeholder when an output grid is missing.

    Args:
        draw: PIL drawing context.
        x: Left coordinate.
        y: Top coordinate.
        width: Placeholder width.
        height: Placeholder height.
        font: Font used for drawing.
    """
    draw.rectangle([x, y, x + width, y + height], fill=MISSING_BG, outline=GRID_LINE)
    draw_centered_text(draw, "?", x, y + height // 2 - 6, width, font)


def draw_pair(
    draw: ImageDraw.ImageDraw,
    example: Example,
    index: int,
    x: int,
    y: int,
    cell_size: int,
    font: ImageFont.ImageFont,
) -> None:
    """Draw one input-output example.

    Args:
        draw: PIL drawing context.
        example: Train or test example.
        index: One-based example index.
        x: Left coordinate.
        y: Top coordinate.
        cell_size: Pixel size for each ARC cell.
        font: Font used for drawing.
    """
    pair_w, _ = pair_size(example, cell_size)

    input_grid = example["input"]
    input_w, input_h = grid_pixel_size(input_grid, cell_size)

    output_grid = example.get("output")
    if output_grid is None:
        output_w, output_h = max(80, input_w), max(40, input_h)
    else:
        output_w, output_h = grid_pixel_size(output_grid, cell_size)

    row_h = max(input_h, output_h)

    draw_centered_text(draw, f"Example {index}", x, y, pair_w, font)

    label_y = y + PAIR_TITLE_H
    grid_y = y + PAIR_TITLE_H + GRID_LABEL_H

    input_x = x
    arrow_x = input_x + input_w + PAIR_GAP
    output_x = arrow_x + ARROW_W + PAIR_GAP
    input_y = grid_y + (row_h - input_h) // 2
    output_y = grid_y + (row_h - output_h) // 2

    draw_centered_text(draw, "input", input_x, label_y, input_w, font)
    draw_centered_text(draw, "output", output_x, label_y, output_w, font)

    draw_grid(draw, input_grid, input_x, input_y, cell_size)
    draw_arrow(draw, arrow_x, grid_y, ARROW_W, row_h)

    if output_grid is None:
        draw_missing_output(draw, output_x, output_y, output_w, output_h, font)
    else:
        draw_grid(draw, output_grid, output_x, output_y, cell_size)
