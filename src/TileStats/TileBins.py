"""Rectangular tile binning and histogram utilities.

Translated from the Wolfram Language paclet "AntonAntonov/TileStats".
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

Number = Union[int, float]
Point = Tuple[Number, Number]
Polygon = Tuple[Point, ...]
BinSize = Union[Number, Tuple[Number, Number]]


# -----------------------------
# Support functions
# -----------------------------

def reference_rectangle() -> List[Point]:
    return [(0, 0), (1, 0), (1, 1), (0, 1)]


def _is_number(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_point(p: object) -> bool:
    return (
        isinstance(p, (list, tuple))
        and len(p) == 2
        and _is_number(p[0])
        and _is_number(p[1])
    )


def _is_matrix_data(data: object) -> bool:
    if not isinstance(data, (list, tuple)):
        return False
    return all(_is_point(p) for p in data)


def _is_rules_data(data: object) -> bool:
    if isinstance(data, Mapping):
        return all(_is_point(k) for k in data.keys())
    if isinstance(data, (list, tuple)):
        return all(
            isinstance(item, (list, tuple))
            and len(item) == 2
            and _is_point(item[0])
            for item in data
        )
    return False


def _normalize_rules_data(
    data: Union[Mapping[Point, Number], Sequence[Tuple[Point, Number]]]
) -> List[Tuple[Point, Number]]:
    if isinstance(data, Mapping):
        return [(tuple(k), data[k]) for k in data.keys()]
    return [(tuple(k), v) for (k, v) in data]


def _as_bin_tuple(bin_size: BinSize) -> Tuple[Number, Number]:
    if isinstance(bin_size, (list, tuple)):
        if len(bin_size) != 2:
            raise ValueError("bin_size must be a number or a 2-tuple")
        return (bin_size[0], bin_size[1])
    return (bin_size, bin_size)


def tile_containing(point: Point) -> Point:
    x, y = point
    return (math.floor(x), math.floor(y))


def nearest_rectangle(point: Point) -> Point:
    return tile_containing(point)


def transform_by_vector(v: Sequence[Point], tr: Point) -> Polygon:
    # WL: Polygon[TranslationTransform[tr][v]]
    return tuple((p[0] + tr[0], p[1] + tr[1]) for p in v)


def rectangle_vertex_distance(bin_size: BinSize, factor: Number) -> List[Point]:
    bx, by = _as_bin_tuple(bin_size)
    return [(factor * bx * x, factor * by * y) for x, y in reference_rectangle()]


def _min_max(points: Sequence[Point]) -> Tuple[Tuple[Number, Number], Tuple[Number, Number]]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), max(xs)), (min(ys), max(ys))


def _group_values(
    pairs: Iterable[Tuple[Point, Number]],
    aggregation_function: Callable[[List[Number]], Number],
) -> Dict[Point, Number]:
    groups: Dict[Point, List[Number]] = defaultdict(list)
    for key, val in pairs:
        groups[key].append(val)
    return {k: aggregation_function(vs) for k, vs in groups.items()}


# -----------------------------
# Tile origin bins
# -----------------------------

def tile_origin_bins(
    data: Union[Sequence[Point], Mapping[Point, Number], Sequence[Tuple[Point, Number]]],
    bin_size: BinSize,
    data_range: Union[None, Tuple[Tuple[Number, Number], Tuple[Number, Number]]] = None,
    *,
    aggregation_function: Callable[[List[Number]], Number] = sum,
) -> Dict[Point, Number]:
    bx, by = _as_bin_tuple(bin_size)

    if _is_matrix_data(data):
        points = [tuple(p) for p in data]  # type: ignore[arg-type]
        if data_range is None:
            data_range = _min_max(points)
        tallied: Dict[Point, int] = defaultdict(int)
        for x, y in points:
            cx, cy = nearest_rectangle((x / bx, y / by))
            center = (bx * cx, by * cy)
            tallied[center] += 1
        return dict(tallied)

    if _is_rules_data(data):
        pairs = _normalize_rules_data(data)  # type: ignore[arg-type]
        if data_range is None:
            data_range = _min_max([p for p, _ in pairs])
        grouped: List[Tuple[Point, Number]] = []
        for (x, y), v in pairs:
            cx, cy = nearest_rectangle((x / bx, y / by))
            origin = (bx * cx, by * cy)
            grouped.append((origin, v))
        return _group_values(grouped, aggregation_function)

    raise ValueError("data must be 2D points or point->value rules")


# -----------------------------
# Tile center bins
# -----------------------------

def tile_center_bins(
    data: Union[Sequence[Point], Mapping[Point, Number], Sequence[Tuple[Point, Number]]],
    bin_size: BinSize,
    data_range: Union[None, Tuple[Tuple[Number, Number], Tuple[Number, Number]]] = None,
    *,
    overlap_factor: Number = 1,
    aggregation_function: Callable[[List[Number]], Number] = sum,
) -> Dict[Point, Number]:
    if overlap_factor <= 0:
        raise ValueError("overlap_factor must be positive")

    vh = rectangle_vertex_distance(bin_size, overlap_factor)
    origins = tile_origin_bins(
        data,
        bin_size,
        data_range,
        aggregation_function=aggregation_function,
    )

    centers: Dict[Point, Number] = {}
    mean_x = sum(p[0] for p in vh) / len(vh)
    mean_y = sum(p[1] for p in vh) / len(vh)
    for origin, val in origins.items():
        centers[(origin[0] + mean_x, origin[1] + mean_y)] = val
    return centers


# -----------------------------
# Tile polygon bins
# -----------------------------

def tile_polygon_bins(
    data: Union[Sequence[Point], Mapping[Point, Number], Sequence[Tuple[Point, Number]]],
    bin_size: BinSize,
    data_range: Union[None, Tuple[Tuple[Number, Number], Tuple[Number, Number]]] = None,
    *,
    overlap_factor: Number = 1,
    aggregation_function: Callable[[List[Number]], Number] = sum,
) -> Dict[Polygon, Number]:
    if overlap_factor <= 0:
        raise ValueError("overlap_factor must be positive")

    vh = rectangle_vertex_distance(bin_size, overlap_factor)
    origins = tile_origin_bins(
        data,
        bin_size,
        data_range,
        aggregation_function=aggregation_function,
    )
    return {transform_by_vector(vh, o): v for o, v in origins.items()}


# -----------------------------
# Tile bins (public)
# -----------------------------

def tile_bins(
    data: Union[Sequence[Point], Mapping[Point, Number], Sequence[Tuple[Point, Number]]],
    bin_size: BinSize,
    data_range: Union[None, Tuple[Tuple[Number, Number], Tuple[Number, Number]]] = None,
    *,
    aggregation_function: Callable[[List[Number]], Number] = sum,
    polygon_keys: bool = True,
    overlap_factor: Number = 1,
) -> Dict[Union[Point, Polygon], Number]:
    bx, by = _as_bin_tuple(bin_size)
    if not (_is_number(bx) and _is_number(by) and bx > 0 and by > 0):
        raise ValueError("bin_size must be positive")

    if overlap_factor <= 0:
        raise ValueError("overlap_factor must be positive")

    if data_range is None:
        if _is_matrix_data(data):
            data_range = _min_max([tuple(p) for p in data])  # type: ignore[arg-type]
        elif _is_rules_data(data):
            pairs = _normalize_rules_data(data)  # type: ignore[arg-type]
            data_range = _min_max([p for p, _ in pairs])

    if polygon_keys:
        return tile_polygon_bins(
            data,
            bin_size,
            data_range,
            overlap_factor=overlap_factor,
            aggregation_function=aggregation_function,
        )

    return tile_center_bins(
        data,
        bin_size,
        data_range,
        overlap_factor=overlap_factor,
        aggregation_function=aggregation_function,
    )


# -----------------------------
# Tile histogram
# -----------------------------

def _rescale(value: Number, vmin: Number, vmax: Number) -> float:
    if vmax == vmin:
        return 0.0
    return float((value - vmin) / (vmax - vmin))


def _default_color_func(t: float) -> Tuple[float, float, float, float]:
    t = max(0.0, min(1.0, t))
    t = math.sqrt(t)
    c0 = (0.92, 0.96, 1.0)
    c1 = (0.12, 0.23, 0.55)
    return (
        c0[0] + (c1[0] - c0[0]) * t,
        c0[1] + (c1[1] - c0[1]) * t,
        c0[2] + (c1[2] - c0[2]) * t,
        1.0,
    )


def _get_color_func(color_function: Union[str, Callable[[float], object], None]):
    if callable(color_function):
        return color_function
    if color_function is None or color_function is True:
        return _default_color_func
    try:
        import matplotlib.cm as cm  # type: ignore

        return cm.get_cmap(color_function)
    except Exception:
        return _default_color_func


def tile_histogram(
    data: Union[Sequence[Point], Mapping[Point, Number], Sequence[Tuple[Point, Number]]],
    bin_size: BinSize,
    data_range: Union[None, Tuple[Tuple[Number, Number], Tuple[Number, Number]]] = None,
    *,
    aggregation_function: Callable[[List[Number]], Number] = sum,
    histogram_type: Union[int, str] = "ColoredPolygons",
    max_tally: Union[Number, None] = None,
    min_tally: Union[Number, None] = None,
    overlap_factor: Number = 1,
    color_function: Union[str, Callable[[float], object], None] = None,
    plot_legends: Union[None, str] = None,
    edge_color: Union[None, str] = None,
    line_width: Union[None, float] = None,
    plot: bool = False,
    **plot_kwargs,
):
    bx, by = _as_bin_tuple(bin_size)
    if not (_is_number(bx) and _is_number(by) and bx > 0 and by > 0):
        raise ValueError("bin_size must be positive")
    if overlap_factor <= 0:
        raise ValueError("overlap_factor must be positive")

    if data_range is None:
        if _is_matrix_data(data):
            data_range = _min_max([tuple(p) for p in data])  # type: ignore[arg-type]
        elif _is_rules_data(data):
            pairs = _normalize_rules_data(data)  # type: ignore[arg-type]
            data_range = _min_max([p for p, _ in pairs])

    vh = rectangle_vertex_distance(bin_size, overlap_factor)
    tally = tile_origin_bins(
        data,
        bin_size,
        data_range,
        aggregation_function=aggregation_function,
    )

    items = list(tally.items())
    if not items:
        return {
            "polygons": [],
            "values": [],
            "colors": [],
            "min": None,
            "max": None,
            "figure": None,
            "ax": None
        }

    values = [v for _, v in items]
    if max_tally is None:
        max_tally = max(values)
    if min_tally is None:
        min_tally = min(values)

    cfunc = _get_color_func(color_function)

    polygons: List[Polygon] = []
    colors: List[object] = []

    for origin, value in items:
        r = _rescale(value, float(min_tally), float(max_tally))

        if histogram_type in (1, "ColoredPolygons"):
            poly = transform_by_vector(vh, origin)
        elif histogram_type in (2, "ProportionalSideSize"):
            poly = transform_by_vector([(r * x, r * y) for x, y in vh], origin)
        elif histogram_type in (3, "ProportionalArea"):
            s = math.sqrt(r)
            poly = transform_by_vector([(s * x, s * y) for x, y in vh], origin)
        else:
            poly = transform_by_vector(vh, origin)

        polygons.append(poly)
        colors.append(cfunc(r))

    fig = None
    if plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            from matplotlib.patches import Polygon as MplPolygon  # type: ignore

            fig, ax = plt.subplots()
            for poly, color in zip(polygons, colors):
                patch = MplPolygon(poly, closed=True, facecolor=color, edgecolor=edge_color, linewidth=line_width)
                ax.add_patch(patch)

            if data_range is not None:
                (xmin, xmax), (ymin, ymax) = data_range
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

            ax.set_aspect("equal", adjustable="box")
            ax.set_frame_on(True)

            if plot_legends == "Automatic":
                try:
                    import matplotlib.cm as cm  # type: ignore
                    import matplotlib.colors as colors_mod  # type: ignore

                    norm = colors_mod.Normalize(vmin=float(min_tally), vmax=float(max_tally))
                    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cfunc), ax=ax)
                except Exception:
                    pass

            if plot_kwargs:
                ax.set(**plot_kwargs)
        except Exception:
            fig = None

    return {
        "polygons": polygons,
        "values": values,
        "colors": colors,
        "min": min_tally,
        "max": max_tally,
        "figure": fig,
        "ax": ax
    }


# -----------------------------
# Tile bins plot
# -----------------------------

def tile_bins_plot(
    bins: Mapping[Polygon, Number],
    *,
    color_function: Union[str, Callable[[float], object], None] = None,
    plot: bool = True,
    **plot_kwargs,
):
    if not bins:
        return {"polygons": [], "values": [], "colors": [], "figure": None}

    cfunc = _get_color_func(color_function)

    items = list(bins.items())
    values = [v for _, v in items]
    max_tally = max(values)

    polygons = [p for p, _ in items]
    colors = [cfunc(v / max_tally) for v in values]

    fig = None
    if plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            from matplotlib.patches import Polygon as MplPolygon  # type: ignore

            fig, ax = plt.subplots()
            for poly, color in zip(polygons, colors):
                patch = MplPolygon(poly, closed=True, facecolor=color, edgecolor="none")
                ax.add_patch(patch)

            ax.set_aspect("equal", adjustable="box")
            ax.set_frame_on(True)

            if plot_kwargs:
                ax.set(**plot_kwargs)
        except Exception:
            fig = None

    return {"polygons": polygons, "values": values, "colors": colors, "figure": fig}


__all__ = [
    "tile_origin_bins",
    "tile_center_bins",
    "tile_polygon_bins",
    "tile_bins",
    "tile_histogram",
    "tile_bins_plot",
]
