#!/usr/bin/env python3
"""Export static figures (PNG/PDF) for LaTeX from sweep/report CSV artifacts.

This script intentionally avoids matplotlib/plotly static backends to keep
exports robust in restricted environments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = _font(42)
FONT_AXIS = _font(26)
FONT_TICK = _font(22)
FONT_SMALL = _font(18)
FONT_LEGEND = _font(22)


def _source_metric_sort_key(source: str, metric: str) -> tuple[int, int, str, str]:
    src = source.strip().lower()
    met = metric.strip().upper()
    src_order = {"fi": 0, "oc": 1, "zipnerf": 2, "zip": 2}
    met_order = {"PSNR": 0, "SSIM": 1}
    return (src_order.get(src, 99), met_order.get(met, 99), src, met)


class Canvas:
    def __init__(self, width: int = 1700, height: int = 1050):
        self.width = width
        self.height = height
        self.image = Image.new("RGB", (width, height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.left = 150
        self.right = 80
        self.top = 120
        # Extra bottom room keeps rotated x-axis labels and bars fully visible.
        self.bottom = 220

    @property
    def x0(self) -> int:
        return self.left

    @property
    def x1(self) -> int:
        return self.width - self.right

    @property
    def y0(self) -> int:
        return self.top

    @property
    def y1(self) -> int:
        return self.height - self.bottom

    @property
    def plot_w(self) -> int:
        return self.x1 - self.x0

    @property
    def plot_h(self) -> int:
        return self.y1 - self.y0

    def save(self, png_path: Path, pdf_path: Path) -> None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(png_path, format="PNG")
        self.image.save(pdf_path, format="PDF", resolution=300.0)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    return x1 - x0, y1 - y0


def _draw_title(canvas: Canvas, title: str) -> None:
    w, h = _text_size(canvas.draw, title, FONT_TITLE)
    canvas.draw.text(((canvas.width - w) // 2, 30), title, fill="black", font=FONT_TITLE)
    canvas.draw.line((canvas.x0, 92, canvas.x1, 92), fill="#d0d0d0", width=2)


def _axis_y(value: float, y_min: float, y_max: float, canvas: Canvas) -> int:
    if y_max <= y_min:
        return canvas.y1
    frac = (value - y_min) / (y_max - y_min)
    frac = max(0.0, min(1.0, frac))
    return int(canvas.y1 - frac * canvas.plot_h)


def _draw_axes(
    canvas: Canvas,
    y_min: float,
    y_max: float,
    y_label: str,
    x_labels: Sequence[str],
    x_rotation: int = 25,
) -> list[int]:
    draw = canvas.draw
    draw.line((canvas.x0, canvas.y0, canvas.x0, canvas.y1), fill="black", width=3)
    draw.line((canvas.x0, canvas.y1, canvas.x1, canvas.y1), fill="black", width=3)

    ticks = 6
    for i in range(ticks + 1):
        frac = i / ticks
        value = y_min + frac * (y_max - y_min)
        y = int(canvas.y1 - frac * canvas.plot_h)
        draw.line((canvas.x0, y, canvas.x1, y), fill="#e6e6e6", width=1)
        label = f"{value:.3f}" if y_max <= 1.0 else f"{value:.2f}"
        tw, th = _text_size(draw, label, FONT_TICK)
        draw.text((canvas.x0 - tw - 14, y - th // 2), label, fill="#444444", font=FONT_TICK)

    ylabel_w, ylabel_h = _text_size(draw, y_label, FONT_AXIS)
    ylabel_img = Image.new("RGBA", (ylabel_h + 20, ylabel_w + 20), (255, 255, 255, 0))
    yd = ImageDraw.Draw(ylabel_img)
    yd.text((10, 10), y_label, fill="black", font=FONT_AXIS)
    ylabel_img = ylabel_img.rotate(90, expand=True)
    canvas.image.paste(ylabel_img, (30, (canvas.height - ylabel_img.height) // 2), ylabel_img)

    n = len(x_labels)
    if n == 1:
        xs = [canvas.x0 + canvas.plot_w // 2]
    else:
        # Category centers (not edges) prevent first/last bars from clipping.
        step = canvas.plot_w / n
        xs = [int(canvas.x0 + (i + 0.5) * step) for i in range(n)]

    for x, label in zip(xs, x_labels):
        draw.line((x, canvas.y1, x, canvas.y1 + 8), fill="black", width=2)
        tw, th = _text_size(draw, label, FONT_TICK)
        if x_rotation == 0:
            draw.text((x - tw // 2, canvas.y1 + 12), label, fill="black", font=FONT_TICK)
        else:
            label_img = Image.new("RGBA", (tw + 8, th + 8), (255, 255, 255, 0))
            ld = ImageDraw.Draw(label_img)
            ld.text((4, 4), label, fill="black", font=FONT_TICK)
            label_img = label_img.rotate(x_rotation, expand=True)
            canvas.image.paste(
                label_img,
                (x - label_img.width // 2, canvas.y1 + 16),
                label_img,
            )
    return xs


def _draw_legend(canvas: Canvas, entries: Sequence[tuple[str, str]], x: int, y: int) -> None:
    draw = canvas.draw
    pad = 12
    box = 20
    cur_y = y
    max_w = 0
    for label, _ in entries:
        tw, _ = _text_size(draw, label, FONT_LEGEND)
        max_w = max(max_w, tw)
    total_h = len(entries) * (box + 10) + pad * 2 - 10
    total_w = box + 12 + max_w + pad * 2
    draw.rounded_rectangle((x, y, x + total_w, y + total_h), radius=10, fill="#ffffff", outline="#cccccc", width=2)
    cur_y += pad
    for label, color in entries:
        draw.rectangle((x + pad, cur_y, x + pad + box, cur_y + box), fill=color, outline=color)
        draw.text((x + pad + box + 10, cur_y - 1), label, fill="black", font=FONT_LEGEND)
        cur_y += box + 10


def _save(canvas: Canvas, out_dir: Path, stem: str) -> None:
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    canvas.save(png_path, pdf_path)


def draw_budget_curve(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path).copy()
    df["budget_per_cluster"] = pd.to_numeric(df["budget_per_cluster"], errors="coerce")
    df = df.dropna(subset=["budget_per_cluster"]).sort_values("budget_per_cluster", kind="mergesort")

    x_labels = [str(int(v)) for v in df["budget_per_cluster"].tolist()]
    pass_vals = pd.to_numeric(df["joint_pass_rate"], errors="coerce").fillna(0.0).tolist()
    lcb_vals = pd.to_numeric(df["joint_pass_rate_lcb"], errors="coerce").fillna(0.0).tolist()
    target = float(pd.to_numeric(df["target_joint_pass_rate"], errors="coerce").dropna().iloc[0])
    recommended_budget = pd.to_numeric(df.loc[df["recommended"] == True, "budget_per_cluster"], errors="coerce")
    recommended = int(recommended_budget.iloc[0]) if not recommended_budget.empty else None

    y_max = max(max(pass_vals), max(lcb_vals), target) * 1.35
    y_max = max(y_max, 0.12)

    canvas = Canvas()
    _draw_title(canvas, "Budget Sweep: Joint Pass Rate and Confidence Bound")
    xs = _draw_axes(canvas, 0.0, y_max, "Pass Probability", x_labels, x_rotation=0)
    draw = canvas.draw

    def points(vals: Sequence[float]) -> list[tuple[int, int]]:
        return [(x, _axis_y(v, 0.0, y_max, canvas)) for x, v in zip(xs, vals)]

    p_pass = points(pass_vals)
    p_lcb = points(lcb_vals)
    draw.line(p_pass, fill="#1f77b4", width=5)
    draw.line(p_lcb, fill="#ff7f0e", width=5)
    for x, y in p_pass:
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#1f77b4")
    for x, y in p_lcb:
        draw.rectangle((x - 6, y - 6, x + 6, y + 6), fill="#ff7f0e")

    y_target = _axis_y(target, 0.0, y_max, canvas)
    draw.line((canvas.x0, y_target, canvas.x1, y_target), fill="#2ca02c", width=3)
    draw.text((canvas.x1 - 300, y_target - 30), f"target={target:.2f}", fill="#2ca02c", font=FONT_SMALL)

    if recommended is not None:
        idx = x_labels.index(str(recommended))
        xr = xs[idx]
        draw.line((xr, canvas.y0, xr, canvas.y1), fill="#d62728", width=2)
        label = f"recommended b={recommended}"
        tw, th = _text_size(draw, label, FONT_SMALL)
        # Keep the label away from the legend box in the top-right corner.
        label_x = xr + 10
        if label_x + tw > canvas.x1 - 360:
            label_x = max(canvas.x0 + 8, xr - tw - 10)
        label_y = canvas.y0 + 10
        draw.rectangle(
            (label_x - 4, label_y - 2, label_x + tw + 4, label_y + th + 2),
            fill="white",
            outline="#d62728",
            width=1,
        )
        draw.text((label_x, label_y), label, fill="#d62728", font=FONT_SMALL)

    draw.text((canvas.width // 2 - 145, canvas.height - 95), "Budget Per Cluster", fill="black", font=FONT_AXIS)
    _draw_legend(
        canvas,
        [
            ("Joint pass rate", "#1f77b4"),
            ("LCB (Wilson)", "#ff7f0e"),
            ("Target pass rate", "#2ca02c"),
        ],
        canvas.x0 + 20,
        canvas.y0 + 20,
    )
    _save(canvas, out_dir, "fig_budget_pass_lcb")


def draw_global_gaps(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path).copy()
    df["label"] = df["csv"].str.upper() + "-" + df["metric"].str.upper()
    labels = df["label"].tolist()
    abs_diff = pd.to_numeric(df["abs_diff"], errors="coerce").fillna(0.0).tolist()
    thresholds = pd.to_numeric(df["threshold"], errors="coerce").fillna(0.0).tolist()
    passes = df["pass"].fillna(False).astype(bool).tolist()
    norm_diff = [((g / t) if t > 0 else 0.0) for g, t in zip(abs_diff, thresholds)]

    y_max = max(max(norm_diff), 1.0) * 1.35
    y_max = max(y_max, 1.25)
    canvas = Canvas()
    _draw_title(canvas, "Global Metric Gaps (Normalized by Threshold)")
    xs = _draw_axes(canvas, 0.0, y_max, "Gap / Threshold", labels, x_rotation=25)
    draw = canvas.draw
    bar_w = max(40, int(canvas.plot_w / (len(xs) * 2.5)))
    y_thr = _axis_y(1.0, 0.0, y_max, canvas)
    draw.line((canvas.x0, y_thr, canvas.x1, y_thr), fill="#1f77b4", width=4)
    draw.text((canvas.x1 - 320, y_thr - 30), "pass boundary = 1.0", fill="#1f77b4", font=FONT_SMALL)

    for x, gap, thr, ratio, ok in zip(xs, abs_diff, thresholds, norm_diff, passes):
        y_gap = _axis_y(ratio, 0.0, y_max, canvas)
        color = "#2ca02c" if ok else "#d62728"
        draw.rectangle((x - bar_w // 2, y_gap, x + bar_w // 2, canvas.y1), fill=color, outline="#333333")
        draw.text((x - bar_w // 2, y_gap - 50), f"{ratio:.2f}x", fill="#222222", font=FONT_SMALL)
        draw.text((x - bar_w // 2, y_gap - 28), f"{gap:.4f}/{thr:.4f}", fill="#666666", font=FONT_SMALL)

    draw.text((canvas.width // 2 - 95, canvas.height - 95), "Metric Group", fill="black", font=FONT_AXIS)
    _draw_legend(
        canvas,
        [
            ("Normalized gap (pass)", "#2ca02c"),
            ("Normalized gap (fail)", "#d62728"),
            ("Pass boundary (=1.0)", "#1f77b4"),
        ],
        canvas.x1 - 390,
        canvas.y0 + 20,
    )
    _save(canvas, out_dir, "fig_global_gaps")


def draw_holdout_gaps(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path).copy()
    df["csv"] = df["csv"].astype(str).str.strip().str.lower()
    df["metric"] = df["metric"].astype(str).str.strip().str.upper()
    categories = sorted(
        {
            (src, met)
            for src, met in zip(df["csv"].tolist(), df["metric"].tolist())
            if (
                ((df["split"] == "tune") & (df["csv"] == src) & (df["metric"] == met)).any()
                and ((df["split"] == "test") & (df["csv"] == src) & (df["metric"] == met)).any()
            )
        },
        key=lambda x: _source_metric_sort_key(x[0], x[1]),
    )
    labels = [f"{src.upper()}-{met}" for src, met in categories]
    if not categories:
        raise ValueError(f"No tune/test metric categories found in {csv_path}")
    tune_vals: list[float] = []
    test_vals: list[float] = []
    thresholds: list[float] = []
    for src, met in categories:
        trow = df[(df["split"] == "tune") & (df["csv"] == src) & (df["metric"] == met)].iloc[0]
        srow = df[(df["split"] == "test") & (df["csv"] == src) & (df["metric"] == met)].iloc[0]
        tune_vals.append(float(trow["abs_diff"]))
        test_vals.append(float(srow["abs_diff"]))
        thresholds.append(float(trow["threshold"]))
    tune_norm = [((v / t) if t > 0 else 0.0) for v, t in zip(tune_vals, thresholds)]
    test_norm = [((v / t) if t > 0 else 0.0) for v, t in zip(test_vals, thresholds)]

    y_max = max(max(tune_norm), max(test_norm), 1.0) * 1.35
    y_max = max(y_max, 1.25)
    canvas = Canvas()
    _draw_title(canvas, "Holdout Validation: Tune/Test Gaps (Normalized)")
    xs = _draw_axes(canvas, 0.0, y_max, "Gap / Threshold", labels, x_rotation=30)
    draw = canvas.draw
    group_w = max(70, int(canvas.plot_w / (len(xs) * 2.1)))
    bar_w = group_w // 2 - 8
    y_thr = _axis_y(1.0, 0.0, y_max, canvas)
    draw.line((canvas.x0, y_thr, canvas.x1, y_thr), fill="#2ca02c", width=4)
    draw.text((canvas.x1 - 320, y_thr - 30), "pass boundary = 1.0", fill="#2ca02c", font=FONT_SMALL)
    for x, vt, vs, vn_tune, vn_test in zip(xs, tune_vals, test_vals, tune_norm, test_norm):
        x_tune = x - bar_w // 2 - 6
        x_test = x + bar_w // 2 + 6
        y_tune = _axis_y(vn_tune, 0.0, y_max, canvas)
        y_test = _axis_y(vn_test, 0.0, y_max, canvas)
        draw.rectangle((x_tune - bar_w // 2, y_tune, x_tune + bar_w // 2, canvas.y1), fill="#1f77b4", outline="#333333")
        draw.rectangle((x_test - bar_w // 2, y_test, x_test + bar_w // 2, canvas.y1), fill="#ff7f0e", outline="#333333")
        draw.text((x_tune - bar_w // 2, y_tune - 28), f"{vt:.4f}", fill="#666666", font=FONT_SMALL)
        draw.text((x_test - bar_w // 2, y_test - 28), f"{vs:.4f}", fill="#666666", font=FONT_SMALL)

    draw.text((canvas.width // 2 - 95, canvas.height - 95), "Metric Group", fill="black", font=FONT_AXIS)
    _draw_legend(
        canvas,
        [
            ("Tune normalized gap", "#1f77b4"),
            ("Test normalized gap", "#ff7f0e"),
            ("Pass boundary (=1.0)", "#2ca02c"),
        ],
        canvas.x1 - 400,
        canvas.y0 + 20,
    )
    _save(canvas, out_dir, "fig_holdout_tune_test")


def draw_refinement(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path).copy()
    df["csv"] = df["csv"].astype(str).str.strip().str.lower()
    df["metric"] = df["metric"].astype(str).str.strip().str.upper()
    categories = sorted(
        {
            (src, met)
            for src, met in zip(df["csv"].tolist(), df["metric"].tolist())
            if (
                ((df["phase"] == "before") & (df["csv"] == src) & (df["metric"] == met)).any()
                and ((df["phase"] == "after") & (df["csv"] == src) & (df["metric"] == met)).any()
            )
        },
        key=lambda x: _source_metric_sort_key(x[0], x[1]),
    )
    labels = [f"{src.upper()}-{met}" for src, met in categories]
    if not categories:
        raise ValueError(f"No before/after metric categories found in {csv_path}")
    before_vals: list[float] = []
    after_vals: list[float] = []
    thresholds: list[float] = []
    for src, met in categories:
        brow = df[(df["phase"] == "before") & (df["csv"] == src) & (df["metric"] == met)].iloc[0]
        arow = df[(df["phase"] == "after") & (df["csv"] == src) & (df["metric"] == met)].iloc[0]
        before_vals.append(float(brow["abs_diff"]))
        after_vals.append(float(arow["abs_diff"]))
        thresholds.append(float(brow["threshold"]))
    before_norm = [((v / t) if t > 0 else 0.0) for v, t in zip(before_vals, thresholds)]
    after_norm = [((v / t) if t > 0 else 0.0) for v, t in zip(after_vals, thresholds)]

    y_max = max(max(before_norm), max(after_norm), 1.0) * 1.35
    y_max = max(y_max, 1.25)
    canvas = Canvas()
    _draw_title(canvas, "Refinement Impact: Before vs After (Normalized)")
    xs = _draw_axes(canvas, 0.0, y_max, "Gap / Threshold", labels, x_rotation=30)
    draw = canvas.draw
    group_w = max(70, int(canvas.plot_w / (len(xs) * 2.1)))
    bar_w = group_w // 2 - 8
    y_thr = _axis_y(1.0, 0.0, y_max, canvas)
    draw.line((canvas.x0, y_thr, canvas.x1, y_thr), fill="#1f77b4", width=4)
    draw.text((canvas.x1 - 320, y_thr - 30), "pass boundary = 1.0", fill="#1f77b4", font=FONT_SMALL)
    for x, vb, va, vn_before, vn_after in zip(xs, before_vals, after_vals, before_norm, after_norm):
        x_before = x - bar_w // 2 - 6
        x_after = x + bar_w // 2 + 6
        y_before = _axis_y(vn_before, 0.0, y_max, canvas)
        y_after = _axis_y(vn_after, 0.0, y_max, canvas)
        draw.rectangle((x_before - bar_w // 2, y_before, x_before + bar_w // 2, canvas.y1), fill="#d62728", outline="#333333")
        draw.rectangle((x_after - bar_w // 2, y_after, x_after + bar_w // 2, canvas.y1), fill="#2ca02c", outline="#333333")
        draw.text((x_before - bar_w // 2, y_before - 28), f"{vb:.4f}", fill="#666666", font=FONT_SMALL)
        draw.text((x_after - bar_w // 2, y_after - 28), f"{va:.4f}", fill="#666666", font=FONT_SMALL)

    draw.text((canvas.width // 2 - 95, canvas.height - 95), "Metric Group", fill="black", font=FONT_AXIS)
    _draw_legend(
        canvas,
        [
            ("Before normalized gap", "#d62728"),
            ("After normalized gap", "#2ca02c"),
            ("Pass boundary (=1.0)", "#1f77b4"),
        ],
        canvas.x1 - 430,
        canvas.y0 + 20,
    )
    _save(canvas, out_dir, "fig_refinement_before_after")


def draw_pipeline(out_dir: Path) -> None:
    canvas = Canvas(width=1850, height=900)
    draw = canvas.draw
    _draw_title(canvas, "BASS Pipeline Overview")

    box_w = 300
    box_h = 140
    y = 280
    xs = [110, 455, 800, 1145, 1490]
    labels = [
        "Raw Scenes +\nPer-scene Features",
        "Normalize +\nCluster Sweep",
        "Budget / kxBudget\nSimulation Sweep",
        "Metric Matching\nValidation",
        "Selected Subset +\nFinal Report Pack",
    ]
    colors = ["#e8f1fa", "#eaf7ea", "#fff3e6", "#fdebec", "#f0ebfa"]
    for x, label, color in zip(xs, labels, colors):
        draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=18, fill=color, outline="#666666", width=3)
        lines = label.split("\n")
        line_y = y + 28
        for line in lines:
            tw, _ = _text_size(draw, line, FONT_AXIS)
            draw.text((x + (box_w - tw) // 2, line_y), line, fill="#222222", font=FONT_AXIS)
            line_y += 42

    for i in range(len(xs) - 1):
        x0 = xs[i] + box_w
        x1 = xs[i + 1]
        yy = y + box_h // 2
        draw.line((x0 + 8, yy, x1 - 16, yy), fill="#444444", width=5)
        draw.polygon([(x1 - 16, yy), (x1 - 30, yy - 8), (x1 - 30, yy + 8)], fill="#444444")

    draw.rounded_rectangle((220, 110, 820, 220), radius=14, fill="#f8f8f8", outline="#bcbcbc", width=2)
    draw.text((245, 140), "Inputs: feats2.csv + OC/FI/ZipNeRF metric tables", fill="#222222", font=FONT_TICK)
    draw.text((245, 175), "Thresholds: |dPSNR|<=0.5, |dSSIM|<=0.01", fill="#222222", font=FONT_TICK)

    draw.rounded_rectangle((1020, 110, 1710, 220), radius=14, fill="#f8f8f8", outline="#bcbcbc", width=2)
    draw.text((1045, 140), "Outputs: manifests, validation JSON/CSV, figures", fill="#222222", font=FONT_TICK)
    draw.text((1045, 175), "Decision: risk-aware (LCB) auto-selection", fill="#222222", font=FONT_TICK)

    _save(canvas, out_dir, "fig_pipeline_overview")


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export static PNG/PDF figures for LaTeX.")
    parser.add_argument(
        "--bundle-dir",
        default="sweep_cluster_k/share_bundle_prism_20260304",
        help="Directory containing prism CSV exports.",
    )
    parser.add_argument(
        "--output-dir",
        default="latex/figures",
        help="Directory to write PNG/PDF figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = Path(args.bundle_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    draw_pipeline(out_dir)
    draw_budget_curve(_require(bundle / "fig1_budget_pass_lcb.csv"), out_dir)
    draw_global_gaps(_require(bundle / "fig2_global_gaps_vs_threshold.csv"), out_dir)
    draw_holdout_gaps(_require(bundle / "fig3_holdout_tune_test_gaps.csv"), out_dir)
    draw_refinement(_require(bundle / "fig4_refinement_before_after.csv"), out_dir)

    print(f"Saved static figures to: {out_dir}")
    for stem in [
        "fig_pipeline_overview",
        "fig_budget_pass_lcb",
        "fig_global_gaps",
        "fig_holdout_tune_test",
        "fig_refinement_before_after",
    ]:
        print(f"- {out_dir / (stem + '.png')}")
        print(f"- {out_dir / (stem + '.pdf')}")


if __name__ == "__main__":
    main()
