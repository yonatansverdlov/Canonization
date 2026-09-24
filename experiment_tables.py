"""Lightweight, dependency-free terminal tables for experiment summaries."""

from __future__ import annotations


def format_mean_std(mean: float, std: float, scale: float = 100.0) -> str:
    """Format a mean and standard deviation, optionally as percent or pp."""
    return f"{mean * scale:.2f} ± {std * scale:.2f}"


def format_table(
    title: str,
    headers: tuple[str, ...] | list[str],
    rows: list[tuple[str, ...]] | list[list[str]],
    *,
    right_align: tuple[int, ...] = (),
) -> str:
    """Create a bordered table without external formatting dependencies."""
    headings = tuple(str(header) for header in headers)
    if not headings:
        raise ValueError("A table needs at least one column")

    values = [tuple(str(value) for value in row) for row in rows]
    if any(len(row) != len(headings) for row in values):
        raise ValueError("Each row must match the number of headers")

    widths = [
        max(len(headings[i]), *(len(row[i]) for row in values))
        for i in range(len(headings))
    ]
    interior = sum(widths) + 3 * len(widths) - 1
    if len(title) > interior:
        widths[-1] += len(title) - interior
        interior = len(title)

    def border(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * (width + 2) for width in widths) + right

    def row_line(cells: tuple[str, ...]) -> str:
        return "│ " + " │ ".join(
            f"{cell:>{widths[i]}}" if i in right_align else f"{cell:<{widths[i]}}"
            for i, cell in enumerate(cells)
        ) + " │"

    lines = [
        "┌" + "─" * interior + "┐",
        "│" + title.center(interior) + "│",
        border("├", "┬", "┤"),
        row_line(headings),
        border("├", "┼", "┤"),
        *(row_line(row) for row in values),
        border("└", "┴", "┘"),
    ]
    return "\n".join(lines)


def print_table(
    title: str,
    headers: tuple[str, ...] | list[str],
    rows: list[tuple[str, ...]] | list[list[str]],
    *,
    right_align: tuple[int, ...] = (),
) -> None:
    print()
    print(format_table(title, headers, rows, right_align=right_align))
