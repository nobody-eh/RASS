#!/usr/bin/env python3
"""Export a publication-style UMAP coverage figure for the 57-D descriptor space.

Visual encodings:
- Background points (all scenes): low-alpha blue.
- Subset points: saturated red with black outline.
- Marker shape: discovered regime (cluster id).
- Thin convex hulls: major regime envelopes.
- Right-side panel: legends, per-cluster coverage bars, and zoom insets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(p, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = _font(42)
FONT_AXIS = _font(24)
FONT_TEXT = _font(20)
FONT_SMALL = _font(18)


# Okabe-Ito friendly palette.
COL_ALL = (0, 114, 178, 65)          # blue (alpha)
COL_SUBSET = (213, 94, 0, 245)       # vermillion
COL_SUBSET_OUT = (20, 20, 20, 230)
HULL_PALETTE = [
    (0, 114, 178, 150),
    (0, 158, 115, 150),
    (230, 159, 0, 150),
    (86, 180, 233, 150),
    (204, 121, 167, 150),
    (240, 228, 66, 170),
]
COL_GRID = (220, 220, 220, 255)
COL_AX = (40, 40, 40, 255)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    return x1 - x0, y1 - y0


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    size: int,
    shape: str,
    fill: Tuple[int, int, int, int],
    outline: Tuple[int, int, int, int] | None = None,
) -> None:
    s = size
    if shape == "circle":
        draw.ellipse((x - s, y - s, x + s, y + s), fill=fill, outline=outline)
    elif shape == "square":
        draw.rectangle((x - s, y - s, x + s, y + s), fill=fill, outline=outline)
    elif shape == "diamond":
        pts = [(x, y - s), (x + s, y), (x, y + s), (x - s, y)]
        draw.polygon(pts, fill=fill, outline=outline)
    elif shape == "triangle_up":
        pts = [(x, y - s), (x + s, y + s), (x - s, y + s)]
        draw.polygon(pts, fill=fill, outline=outline)
    elif shape == "triangle_down":
        pts = [(x - s, y - s), (x + s, y - s), (x, y + s)]
        draw.polygon(pts, fill=fill, outline=outline)
    elif shape == "cross":
        draw.line((x - s, y, x + s, y), fill=fill, width=2)
        draw.line((x, y - s, x, y + s), fill=fill, width=2)
    else:
        draw.ellipse((x - s, y - s, x + s, y + s), fill=fill, outline=outline)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotonic chain convex hull. Input shape: (N,2), output closed polygon."""
    if points.shape[0] <= 2:
        return points
    pts = np.unique(points.astype(float), axis=0)
    if pts.shape[0] <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.vstack(lower[:-1] + upper[:-1])
    return hull


def _connected_components_by_radius(points: np.ndarray, radius: float) -> List[np.ndarray]:
    """Simple radius-graph components, returns list of index arrays."""
    n = points.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [np.array([0], dtype=int)]
    r2 = float(radius * radius)
    unvisited = set(range(n))
    comps: List[np.ndarray] = []
    while unvisited:
        seed = unvisited.pop()
        comp = [seed]
        stack = [seed]
        while stack:
            i = stack.pop()
            if not unvisited:
                continue
            rem = np.fromiter(unvisited, dtype=int)
            d2 = ((points[rem] - points[i]) ** 2).sum(axis=1)
            neigh = rem[d2 <= r2]
            for j in neigh.tolist():
                if j in unvisited:
                    unvisited.remove(j)
                    stack.append(j)
                    comp.append(j)
        comps.append(np.array(comp, dtype=int))
    return comps


def _draw_axes_and_grid(
    draw: ImageDraw.ImageDraw,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> None:
    draw.line((x0, y1, x1, y1), fill=COL_AX, width=2)
    draw.line((x0, y0, x0, y1), fill=COL_AX, width=2)
    ticks = 5
    for i in range(ticks + 1):
        fx = i / ticks
        fy = i / ticks
        xt = int(x0 + fx * (x1 - x0))
        yt = int(y1 - fy * (y1 - y0))
        draw.line((xt, y0, xt, y1), fill=COL_GRID, width=1)
        draw.line((x0, yt, x1, yt), fill=COL_GRID, width=1)
        xv = xmin + fx * (xmax - xmin)
        yv = ymin + fy * (ymax - ymin)
        draw.text((xt - 18, y1 + 8), f"{xv:.1f}", fill=(90, 90, 90, 255), font=FONT_SMALL)
        draw.text((x0 - 66, yt - 10), f"{yv:.1f}", fill=(90, 90, 90, 255), font=FONT_SMALL)


def _select_region_bounds(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    pad_frac: float = 0.2,
) -> Tuple[float, float, float, float]:
    if mask.sum() == 0:
        return float(x.min()), float(x.max()), float(y.min()), float(y.max())
    xs = x[mask]
    ys = y[mask]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    dx = max(1e-6, xmax - xmin)
    dy = max(1e-6, ymax - ymin)
    return (
        xmin - pad_frac * dx,
        xmax + pad_frac * dx,
        ymin - pad_frac * dy,
        ymax + pad_frac * dy,
    )


def _draw_zoom_panel(
    draw: ImageDraw.ImageDraw,
    panel: Tuple[int, int, int, int],
    df: pd.DataFrame,
    region_bounds: Tuple[float, float, float, float],
    title: str,
    cluster_shape: Dict[int, str],
) -> None:
    px0, py0, px1, py1 = panel
    draw.rounded_rectangle((px0, py0, px1, py1), radius=8, fill=(255, 255, 255, 240), outline=(170, 170, 170, 255), width=2)
    draw.text((px0 + 10, py0 + 8), title, fill=(30, 30, 30, 255), font=FONT_SMALL)
    ix0, iy0, ix1, iy1 = px0 + 12, py0 + 34, px1 - 12, py1 - 12
    draw.line((ix0, iy1, ix1, iy1), fill=(130, 130, 130, 255), width=1)
    draw.line((ix0, iy0, ix0, iy1), fill=(130, 130, 130, 255), width=1)

    rx0, rx1, ry0, ry1 = region_bounds
    reg = df[
        (df["umap_x"] >= rx0) & (df["umap_x"] <= rx1) &
        (df["umap_y"] >= ry0) & (df["umap_y"] <= ry1)
    ].copy()
    if reg.empty:
        draw.text((ix0 + 8, iy0 + 12), "No points in region", fill=(120, 120, 120, 255), font=FONT_SMALL)
        return

    # Keep equal aspect ratio in inset (avoid shape distortion).
    data_w = max(1e-9, rx1 - rx0)
    data_h = max(1e-9, ry1 - ry0)
    view_w = max(1, ix1 - ix0)
    view_h = max(1, iy1 - iy0)
    scale = min(view_w / data_w, view_h / data_h)
    draw_w = data_w * scale
    draw_h = data_h * scale
    off_x = ix0 + (view_w - draw_w) / 2.0
    off_y = iy0 + (view_h - draw_h) / 2.0

    def map_xy(vx: float, vy: float) -> Tuple[int, int]:
        xx = int(off_x + (vx - rx0) * scale)
        yy = int(off_y + draw_h - (vy - ry0) * scale)
        return xx, yy

    for row in reg.itertuples(index=False):
        xx, yy = map_xy(float(row.umap_x), float(row.umap_y))
        _draw_marker(draw, xx, yy, 3, cluster_shape[int(row.cluster)], COL_ALL)
    sub = reg[reg["is_subset"]]
    for row in sub.itertuples(index=False):
        xx, yy = map_xy(float(row.umap_x), float(row.umap_y))
        _draw_marker(draw, xx, yy, 6, cluster_shape[int(row.cluster)], COL_SUBSET, outline=COL_SUBSET_OUT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export UMAP coverage figure.")
    p.add_argument("--features-csv", type=Path, default=Path("feats_normalized.csv"))
    p.add_argument(
        "--cluster-mapping-csv",
        type=Path,
        default=Path("sweep_cluster_k/k_6/clustered_scenes_k6_dish_cluster_mapping.csv"),
    )
    p.add_argument(
        "--subset-manifest-csv",
        type=Path,
        default=Path("sweep_cluster_k/budget_sweep_k6_auto_v2/recommended_subset.csv"),
    )
    p.add_argument("--dish-col", default="dish_id")
    p.add_argument("--cluster-col", default="cluster")
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--width", type=int, default=2100)
    p.add_argument("--height", type=int, default=1200)
    p.add_argument(
        "--show-cluster-hulls",
        action="store_true",
        help="Draw cluster hull lines (can add clutter).",
    )
    p.add_argument(
        "--hide-zoom-links",
        action="store_true",
        help="Disable zoom region boxes + connector lines.",
    )
    p.add_argument("--out-png", type=Path, default=Path("latex/figures/fig_umap_coverage_57d.png"))
    p.add_argument("--out-pdf", type=Path, default=Path("latex/figures/fig_umap_coverage_57d.pdf"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import umap  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "umap-learn is required. Run this script in the environment that has `umap` installed."
        ) from exc

    feats = pd.read_csv(args.features_csv)
    mapping = pd.read_csv(args.cluster_mapping_csv)
    subset = pd.read_csv(args.subset_manifest_csv)

    if args.dish_col not in feats.columns:
        raise ValueError(f"{args.features_csv}: missing `{args.dish_col}` column.")
    if args.dish_col not in mapping.columns or args.cluster_col not in mapping.columns:
        raise ValueError(
            f"{args.cluster_mapping_csv}: expected `{args.dish_col}` and `{args.cluster_col}`."
        )
    if args.dish_col not in subset.columns:
        raise ValueError(f"{args.subset_manifest_csv}: missing `{args.dish_col}` column.")

    df = feats.merge(
        mapping[[args.dish_col, args.cluster_col]],
        on=args.dish_col,
        how="left",
    )
    df = df.dropna(subset=[args.cluster_col]).copy()
    df[args.cluster_col] = pd.to_numeric(df[args.cluster_col], errors="coerce")
    df = df.dropna(subset=[args.cluster_col]).copy()
    df[args.cluster_col] = df[args.cluster_col].astype(int)
    if args.cluster_col != "cluster":
        df = df.rename(columns={args.cluster_col: "cluster"})
    if args.dish_col != "dish_id":
        df = df.rename(columns={args.dish_col: "dish_id"})

    numeric_cols = [c for c in df.columns if c not in ("dish_id", "cluster")]
    x = df[numeric_cols].to_numpy(dtype=float)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="euclidean",
        random_state=args.random_seed,
    )
    emb = reducer.fit_transform(x)
    df["umap_x"] = emb[:, 0]
    df["umap_y"] = emb[:, 1]

    subset_col = args.dish_col if args.dish_col in subset.columns else "dish_id"
    subset_ids = set(subset[subset_col].astype(str))
    df["is_subset"] = df["dish_id"].astype(str).isin(subset_ids)

    width = args.width
    height = args.height
    img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    left, right, top, bottom = 120, 560, 120, 170
    x0, x1 = left, width - right
    y0, y1 = top, height - bottom

    xmin, xmax = float(df["umap_x"].min()), float(df["umap_x"].max())
    ymin, ymax = float(df["umap_y"].min()), float(df["umap_y"].max())
    xpad = (xmax - xmin) * 0.03 if xmax > xmin else 1.0
    ypad = (ymax - ymin) * 0.03 if ymax > ymin else 1.0
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad

    def map_xy(px: float, py: float) -> Tuple[int, int]:
        fx = 0.5 if xmax <= xmin else (px - xmin) / (xmax - xmin)
        fy = 0.5 if ymax <= ymin else (py - ymin) / (ymax - ymin)
        xi = int(x0 + fx * (x1 - x0))
        yi = int(y1 - fy * (y1 - y0))
        return xi, yi

    _draw_axes_and_grid(draw, x0, x1, y0, y1, xmin, xmax, ymin, ymax)

    cluster_ids = sorted(df["cluster"].unique().tolist())
    shape_cycle = ["circle", "square", "diamond", "triangle_up", "triangle_down", "cross"]
    cluster_shape: Dict[int, str] = {
        int(c): shape_cycle[i % len(shape_cycle)] for i, c in enumerate(cluster_ids)
    }

    # Draw all scenes first.
    for row in df.itertuples(index=False):
        xi, yi = map_xy(float(row.umap_x), float(row.umap_y))
        shape = cluster_shape[int(row.cluster)]
        _draw_marker(draw, xi, yi, 3, shape, COL_ALL)

    # Optional hull overlays (can clutter camera-ready figures).
    if args.show_cluster_hulls:
        for idx, cid in enumerate(cluster_ids):
            pts = df[df["cluster"] == cid][["umap_x", "umap_y"]].to_numpy(dtype=float)
            if pts.shape[0] < 4:
                continue
            d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
            np.fill_diagonal(d2, np.inf)
            nn = np.sqrt(np.min(d2, axis=1))
            radius = max(float(np.median(nn) * 3.0), 0.35)
            comps = _connected_components_by_radius(pts, radius=radius)
            col = HULL_PALETTE[idx % len(HULL_PALETTE)]
            for comp in comps:
                if comp.shape[0] < 25:
                    continue
                hull = _convex_hull(pts[comp])
                if hull.shape[0] < 3:
                    continue
                poly = [map_xy(float(px), float(py)) for px, py in hull]
                draw.line(poly + [poly[0]], fill=col, width=2)

    # Draw subset overlay.
    sub_df = df[df["is_subset"]].copy()
    for row in sub_df.itertuples(index=False):
        xi, yi = map_xy(float(row.umap_x), float(row.umap_y))
        shape = cluster_shape[int(row.cluster)]
        _draw_marker(draw, xi, yi, 6, shape, COL_SUBSET, outline=COL_SUBSET_OUT)

    # Mean nearest distance (all -> subset) in UMAP space.
    mean_nn = float("nan")
    if len(sub_df) > 0:
        all_xy = df[["umap_x", "umap_y"]].to_numpy(dtype=float)
        sub_xy = sub_df[["umap_x", "umap_y"]].to_numpy(dtype=float)
        d2 = ((all_xy[:, None, :] - sub_xy[None, :, :]) ** 2).sum(axis=2)
        mean_nn = float(np.sqrt(d2.min(axis=1)).mean())

    title = "UMAP Coverage of the 57-D Descriptor Space"
    tw, th = _text_size(draw, title, FONT_TITLE)
    draw.text(((width - tw) // 2, 30), title, fill=(20, 20, 20, 255), font=FONT_TITLE)

    subtitle1 = (
        f"All scenes: {len(df)} | Selected subset: {len(sub_df)} | Regimes: {len(cluster_ids)} (marker shape)"
    )
    sw, _ = _text_size(draw, subtitle1, FONT_TEXT)
    draw.text(((width - sw) // 2, 78), subtitle1, fill=(70, 70, 70, 255), font=FONT_TEXT)
    subtitle2 = f"Mean nearest selected-scene distance in UMAP space: {mean_nn:.3f}"
    sw2, _ = _text_size(draw, subtitle2, FONT_SMALL)
    draw.text(((width - sw2) // 2, 102), subtitle2, fill=(90, 90, 90, 255), font=FONT_SMALL)

    # Keep x-axis label close to the axis line, independent of canvas size.
    draw.text(((x0 + x1) // 2 - 45, y1 + 22), "UMAP-1", fill=(30, 30, 30, 255), font=FONT_AXIS)
    ylab = "UMAP-2"
    ylab_img = Image.new("RGBA", (120, 40), (255, 255, 255, 0))
    ydraw = ImageDraw.Draw(ylab_img, "RGBA")
    ydraw.text((0, 0), ylab, fill=(30, 30, 30, 255), font=FONT_AXIS)
    ylab_img = ylab_img.rotate(90, expand=True)
    img.paste(ylab_img, (25, (height - ylab_img.height) // 2), ylab_img)

    # Right panel start (outside main axis).
    rp_x0, rp_x1 = x1 + 30, width - 35

    # Point legend (outside axes).
    lg_x, lg_y = rp_x0, y0 + 12
    draw.rounded_rectangle((lg_x, lg_y, lg_x + (rp_x1 - rp_x0), lg_y + 120), radius=10, fill=(255, 255, 255, 245), outline=(180, 180, 180, 255), width=2)
    _draw_marker(draw, lg_x + 20, lg_y + 30, 6, "circle", COL_ALL)
    draw.text((lg_x + 38, lg_y + 20), "All scenes (blue)", fill=(20, 20, 20, 255), font=FONT_TEXT)
    _draw_marker(draw, lg_x + 20, lg_y + 80, 6, "circle", COL_SUBSET, outline=COL_SUBSET_OUT)
    draw.text((lg_x + 38, lg_y + 70), "Selected subset (red)", fill=(20, 20, 20, 255), font=FONT_TEXT)

    # Coverage ratio bar chart per cluster.
    total_counts = df["cluster"].value_counts().sort_index()
    sel_counts = sub_df["cluster"].value_counts().sort_index()
    chart_x0, chart_x1 = rp_x0, rp_x1
    chart_y0, chart_y1 = lg_y + 140, lg_y + 360
    draw.rounded_rectangle((chart_x0, chart_y0, chart_x1, chart_y1), radius=10, fill=(255, 255, 255, 245), outline=(180, 180, 180, 255), width=2)
    draw.text((chart_x0 + 10, chart_y0 + 8), "Subset coverage ratio per cluster", fill=(30, 30, 30, 255), font=FONT_SMALL)
    n = len(cluster_ids)
    row_h = max(18, int((chart_y1 - chart_y0 - 44) / max(1, n)))
    bar_left = chart_x0 + 95
    bar_right = chart_x1 - 12
    for i, cid in enumerate(cluster_ids):
        yy = chart_y0 + 36 + i * row_h
        tot = int(total_counts.get(cid, 0))
        sel = int(sel_counts.get(cid, 0))
        ratio = (sel / tot) if tot > 0 else 0.0
        draw.text((chart_x0 + 10, yy - 2), f"C{cid}", fill=(30, 30, 30, 255), font=FONT_SMALL)
        draw.rectangle((bar_left, yy, bar_right, yy + 12), fill=(236, 236, 236, 255), outline=(210, 210, 210, 255))
        bw = int((bar_right - bar_left) * ratio)
        draw.rectangle((bar_left, yy, bar_left + bw, yy + 12), fill=(0, 158, 115, 230), outline=(0, 158, 115, 230))
        draw.text((bar_right - 92, yy - 2), f"{sel}/{tot}", fill=(70, 70, 70, 255), font=FONT_SMALL)

    # Inset regions on main plot.
    left_mask = df["umap_x"].to_numpy() <= np.quantile(df["umap_x"].to_numpy(), 0.08)
    bottom_mask = df["umap_y"].to_numpy() <= np.quantile(df["umap_y"].to_numpy(), 0.03)
    left_bounds = _select_region_bounds(df["umap_x"].to_numpy(), df["umap_y"].to_numpy(), left_mask, pad_frac=0.12)
    bottom_bounds = _select_region_bounds(df["umap_x"].to_numpy(), df["umap_y"].to_numpy(), bottom_mask, pad_frac=0.15)

    def draw_region_rect(bounds: Tuple[float, float, float, float], col: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        bx0, bx1, by0, by1 = bounds
        p0 = map_xy(bx0, by0)
        p1 = map_xy(bx1, by1)
        rx0, rx1 = min(p0[0], p1[0]), max(p0[0], p1[0])
        ry0, ry1 = min(p0[1], p1[1]), max(p0[1], p1[1])
        draw.rectangle((rx0, ry0, rx1, ry1), outline=col, width=2)
        return rx0, ry0, rx1, ry1

    show_zoom_links = not args.hide_zoom_links
    left_rect = None
    bot_rect = None
    if show_zoom_links:
        left_rect = draw_region_rect(left_bounds, (0, 158, 115, 220))
        bot_rect = draw_region_rect(bottom_bounds, (204, 121, 167, 220))

    # Insets.
    ins1 = (rp_x0, chart_y1 + 18, rp_x1, chart_y1 + 268)
    ins2 = (rp_x0, chart_y1 + 282, rp_x1, chart_y1 + 532)
    _draw_zoom_panel(draw, ins1, df, left_bounds, "Zoom A: sparse left islands", cluster_shape)
    _draw_zoom_panel(draw, ins2, df, bottom_bounds, "Zoom B: tiny bottom island", cluster_shape)

    # Optional connectors main->insets, with colors matched to region boxes.
    if show_zoom_links and left_rect is not None and bot_rect is not None:
        draw.line(
            (left_rect[2], (left_rect[1] + left_rect[3]) // 2, ins1[0], (ins1[1] + ins1[3]) // 2),
            fill=(0, 158, 115, 200),
            width=2,
        )
        draw.line(
            (bot_rect[2], (bot_rect[1] + bot_rect[3]) // 2, ins2[0], (ins2[1] + ins2[3]) // 2),
            fill=(204, 121, 167, 200),
            width=2,
        )

    # Regime marker legend (bottom-left to avoid covering dense clusters).
    rg_x = x0 + 20
    rg_h = 40 + 24 * len(cluster_ids)
    rg_y = y1 - rg_h - 20
    rg_h = 40 + 24 * len(cluster_ids)
    draw.rounded_rectangle((rg_x, rg_y, rg_x + 230, rg_y + rg_h), radius=10, fill=(255, 255, 255, 235), outline=(180, 180, 180, 255), width=2)
    draw.text((rg_x + 12, rg_y + 10), "Regime markers", fill=(20, 20, 20, 255), font=FONT_TEXT)
    cy = rg_y + 42
    for cid in cluster_ids:
        shape = cluster_shape[int(cid)]
        _draw_marker(draw, rg_x + 16, cy + 8, 5, shape, (60, 60, 60, 255))
        draw.text((rg_x + 32, cy), f"Cluster {cid}", fill=(30, 30, 30, 255), font=FONT_SMALL)
        cy += 24

    settings = f"UMAP settings: n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, seed={args.random_seed}"
    draw.text((rp_x0 + 6, y1 + 10), settings, fill=(95, 95, 95, 255), font=FONT_SMALL)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(args.out_png, format="PNG")
    img.convert("RGB").save(args.out_pdf, format="PDF", resolution=300.0)
    print(f"Saved: {args.out_png}")
    print(f"Saved: {args.out_pdf}")


if __name__ == "__main__":
    main()
