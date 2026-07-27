#!/usr/bin/env python3
"""Analyze whether a selected subset preserves cross-method rankings.

This script loads per-scene method metrics from CSV/XLSX files, computes
method-level benchmark rankings on:
1) the full scene set shared by all methods
2) the selected subset intersected with that shared scene set

It then reports ranking-fidelity statistics such as Spearman rho, Kendall tau,
top-1 agreement, and pairwise ordering agreement.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


LOGGER = logging.getLogger("cross_method_ranking_fidelity")

METRIC_DIRECTIONS: Dict[str, str] = {
    "psnr": "higher",
    "ssim": "higher",
    "lpips": "lower",
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    path: Path
    sheet_name: Optional[str] = None


@dataclass
class MethodTable:
    name: str
    path: Path
    df: pd.DataFrame


def _normalize_col_name(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(col).lower())


def _pick_first(
    norm_to_orig: Dict[str, str],
    candidates: Sequence[str],
) -> Optional[str]:
    for cand in candidates:
        if cand in norm_to_orig:
            return norm_to_orig[cand]
    return None


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(raw: str, base_dir: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _read_table(path: Path, sheet_name: Optional[str]) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        if sheet_name is not None:
            return pd.read_excel(path, sheet_name=sheet_name)
        xl = pd.ExcelFile(path)
        if not xl.sheet_names:
            raise ValueError(f"No sheets found in workbook: {path}")
        return xl.parse(xl.sheet_names[0])
    raise ValueError(f"Unsupported metrics file type: {path}")


def _canonicalize_method_table(spec: MethodSpec) -> MethodTable:
    raw = _read_table(spec.path, spec.sheet_name)
    if raw.empty:
        raise ValueError(f"{spec.name}: metrics table is empty: {spec.path}")

    norm_to_orig: Dict[str, str] = {}
    for col in raw.columns:
        norm = _normalize_col_name(col)
        if norm not in norm_to_orig:
            norm_to_orig[norm] = str(col)

    id_col = _pick_first(
        norm_to_orig,
        ("dishid", "experimentname", "sceneid", "id"),
    )
    if id_col is None:
        raise ValueError(
            f"{spec.name}: unable to find scene identifier column in {list(raw.columns)}"
        )

    metric_map = {
        "psnr": _pick_first(
            norm_to_orig,
            ("psnravgfrommse", "psnravgmse", "psnr", "pnsr", "psnrmean"),
        ),
        "ssim": _pick_first(norm_to_orig, ("ssim", "ssimmean")),
        "lpips": _pick_first(norm_to_orig, ("lpips", "lpipsmean")),
    }

    out = pd.DataFrame()
    out["scene_id"] = raw[id_col].astype(str).str.strip()
    out = out[out["scene_id"] != ""].copy()
    for metric_name, src_col in metric_map.items():
        if src_col is None:
            out[metric_name] = np.nan
        else:
            out[metric_name] = pd.to_numeric(raw[src_col], errors="coerce")

    out = (
        out.groupby("scene_id", as_index=False)[["psnr", "ssim", "lpips"]]
        .mean(numeric_only=True)
        .sort_values("scene_id", kind="mergesort")
        .reset_index(drop=True)
    )
    return MethodTable(name=spec.name, path=spec.path, df=out)


def _load_subset_ids(path: Path) -> Set[str]:
    df = pd.read_csv(path)
    norm_to_orig = {_normalize_col_name(c): c for c in df.columns}
    id_col = _pick_first(norm_to_orig, ("dishid", "experimentname", "sceneid", "id"))
    if id_col is None:
        raise ValueError(f"Unable to find subset identifier column in {path}")
    return set(df[id_col].astype(str).str.strip())


def _compute_pairwise_ordering_agreement(
    score_df: pd.DataFrame,
    direction: str,
) -> float:
    values = score_df[["method", "full_mean", "subset_mean"]].to_dict("records")
    total = 0
    agreed = 0
    higher_is_better = direction == "higher"
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            a = values[i]
            b = values[j]
            full_diff = float(a["full_mean"]) - float(b["full_mean"])
            subset_diff = float(a["subset_mean"]) - float(b["subset_mean"])
            if not higher_is_better:
                full_diff *= -1.0
                subset_diff *= -1.0
            full_sign = int(np.sign(full_diff))
            subset_sign = int(np.sign(subset_diff))
            total += 1
            if full_sign == subset_sign:
                agreed += 1
    return float(agreed / total) if total > 0 else float("nan")


def _top_method_names(score_df: pd.DataFrame, metric: str) -> Set[str]:
    direction = METRIC_DIRECTIONS[metric]
    if direction == "higher":
        best_value = float(score_df["score"].max())
        mask = np.isclose(score_df["score"], best_value)
    else:
        best_value = float(score_df["score"].min())
        mask = np.isclose(score_df["score"], best_value)
    return set(score_df.loc[mask, "method"].astype(str))


def _format_metric_label(metric: str) -> str:
    return metric.upper() if metric != "lpips" else "LPIPS"


def _format_method_label(method: str) -> str:
    labels = {
        "instant-ngp-fi": "Instant-NGP",
        "feature-splatting": "Feature\nSplatting",
        "zipnerf": "ZipNeRF",
    }
    return labels.get(str(method), str(method))


def _latex_escape(text: str) -> str:
    out = str(text)
    out = out.replace("\\", "\\textbackslash{}")
    out = out.replace("_", "\\_")
    out = out.replace("%", "\\%")
    out = out.replace("&", "\\&")
    out = out.replace("#", "\\#")
    return out


def _write_latex_table(path: Path, summary_df: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\hline",
        "Subset & Metric & Size & Spearman $\\rho$ & Kendall $\\tau$ & Top-1 & Pairwise \\\\",
        "\\hline",
    ]
    for row in summary_df.to_dict("records"):
        lines.append(
            "{} & {} & {} & {:.3f} & {:.3f} & {} & {:.3f} \\\\".format(
                _latex_escape(row["subset_method"]),
                _latex_escape(str(row["ranking_metric"]).upper()),
                int(row["subset_size"]),
                float(row["spearman_rho"]),
                float(row["kendall_tau"]),
                "Yes" if bool(row["top_1_agreement"]) else "No",
                float(row["pairwise_ordering_agreement"]),
            )
        )
    lines += ["\\hline", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_interpretation(
    path: Path,
    subset_name: str,
    subset_method: str,
    subset_manifest: Path,
    nominal_subset_size: int,
    metric_coverage_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    lines = [
        "# Cross-Method Ranking Fidelity",
        "",
        f"- Subset name: `{subset_name}`",
        f"- Subset method: `{subset_method}`",
        f"- Subset manifest: `{subset_manifest}`",
        f"- Nominal subset size: **{nominal_subset_size}**",
        "",
        "## Effective Scene Coverage",
        "",
        "| ranking_metric | shared_full_size | shared_subset_size | num_methods |",
        "|---|---:|---:|---:|",
    ]
    for row in metric_coverage_df.to_dict("records"):
        lines.append(
            f"| {str(row['ranking_metric']).upper()} | {int(row['shared_full_size'])} | "
            f"{int(row['shared_subset_size'])} | {int(row['num_methods'])} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
    ]

    for row in summary_df.to_dict("records"):
        metric = str(row["ranking_metric"]).upper()
        rho = float(row["spearman_rho"])
        tau = float(row["kendall_tau"])
        pairwise = float(row["pairwise_ordering_agreement"])
        top1 = "preserved" if bool(row["top_1_agreement"]) else "changed"
        lines.append(
            f"- {metric}: top-1 was **{top1}**, with Spearman rho = **{rho:.3f}**, "
            f"Kendall tau = **{tau:.3f}**, and pairwise ordering agreement = **{pairwise:.3f}**."
        )

    if not metric_coverage_df.empty:
        min_subset = int(metric_coverage_df["shared_subset_size"].min())
        min_full = int(metric_coverage_df["shared_full_size"].min())
        lines += [
            "",
            "## Assumption",
            "",
            "- Rankings are computed on the scene intersection shared by all included methods.",
            f"- With the provided files, that leaves **{min_full}** shared full-set scenes and "
            f"as few as **{min_subset}** shared subset scenes, so the result is reproducible but coverage-limited.",
        ]

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_rank_figure(
    path: Path,
    score_df: pd.DataFrame,
    metrics: Sequence[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except Exception as exc:  # pragma: no cover - optional dependency behavior
        LOGGER.warning("Skipping figure export because matplotlib is unavailable: %s", exc)
        return

    def _fmt_score(metric: str, value: float) -> str:
        return f"{value:.3f}" if metric == "psnr" else f"{value:.4f}"

    def _fmt_delta(metric: str, full: float, subset: float) -> str:
        delta = subset - full
        return f"{delta:+.3f}" if metric == "psnr" else f"{delta:+.4f}"

    def _rank_shift(full_rank: float, subset_rank: float) -> str:
        shift = int(round(full_rank - subset_rank))
        if shift == 0:
            return "same"
        return f"+{shift}" if shift > 0 else str(shift)

    method_names = list(dict.fromkeys(score_df["method"].astype(str).tolist()))
    color_map = {
        name: color
        for name, color in zip(
            method_names,
            ["#0f766e", "#2563eb", "#b45309", "#7c3aed", "#be185d", "#475569"],
        )
    }

    n_metrics = len(metrics)
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
        }
    )

    fig, axes = plt.subplots(1, n_metrics, figsize=(5.35 * n_metrics, 4.65), squeeze=False)
    axes_list = axes[0]
    full_n = int(score_df["full_scene_count"].dropna().min()) if "full_scene_count" in score_df else None
    subset_n = int(score_df["subset_scene_count"].dropna().min()) if "subset_scene_count" in score_df else None
    all_stable = bool(np.allclose(score_df["full_rank"], score_df["subset_rank"]))
    text = "#0f172a"
    muted = "#64748b"
    grid = "#e2e8f0"
    good = "#047857"
    row_fill = "#f8fafc"
    header_fill = "#eef6ff"

    for ax, metric in zip(axes_list, metrics):
        sub = score_df[score_df["ranking_metric"] == metric].copy()
        sub = sub.sort_values("full_rank", kind="mergesort").reset_index(drop=True)
        n_rows = len(sub)
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        panel = FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            transform=ax.transAxes,
            linewidth=1.0,
            edgecolor=grid,
            facecolor="#ffffff",
            clip_on=False,
            zorder=-10,
        )
        ax.add_patch(panel)

        ax.text(0.04, 0.93, _format_metric_label(metric), ha="left", va="center", fontsize=16, fontweight="semibold", color=text)
        ax.text(
            0.96,
            0.93,
            "ordering preserved" if np.allclose(sub["full_rank"], sub["subset_rank"]) else "ordering changed",
            ha="right",
            va="center",
            fontsize=9.3,
            color=good if np.allclose(sub["full_rank"], sub["subset_rank"]) else "#b91c1c",
            fontweight="semibold",
        )

        header = FancyBboxPatch(
            (0.035, 0.79),
            0.93,
            0.08,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            transform=ax.transAxes,
            linewidth=0,
            facecolor=header_fill,
            zorder=-2,
        )
        ax.add_patch(header)

        ax.text(0.08, 0.83, "Rank / Method", ha="left", va="center", fontsize=8.6, color=muted, fontweight="semibold")
        ax.text(0.54, 0.83, "Full", ha="right", va="center", fontsize=8.6, color=muted, fontweight="semibold")
        ax.text(0.70, 0.83, "RASS-48", ha="right", va="center", fontsize=8.6, color=muted, fontweight="semibold")
        ax.text(0.83, 0.83, "Delta", ha="right", va="center", fontsize=8.6, color=muted, fontweight="semibold")
        ax.text(0.945, 0.83, "Rank", ha="center", va="center", fontsize=8.6, color=muted, fontweight="semibold")

        row_top = 0.68
        row_gap = 0.20 if n_rows <= 3 else 0.58 / max(n_rows - 1, 1)
        row_h = min(0.145, row_gap * 0.72)

        for idx, row in enumerate(sub.to_dict("records")):
            name = str(row["method"])
            y = row_top - idx * row_gap
            color = color_map.get(name, "#444444")

            label = _format_method_label(name)
            full_mean = float(row["full_mean"])
            subset_mean = float(row["subset_mean"])
            full_value = _fmt_score(metric, float(row["full_mean"]))
            subset_value = _fmt_score(metric, float(row["subset_mean"]))

            row_bg = FancyBboxPatch(
                (0.035, y - row_h / 2),
                0.93,
                row_h,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                transform=ax.transAxes,
                linewidth=0.8,
                edgecolor="#edf2f7",
                facecolor=row_fill,
                zorder=-3,
            )
            ax.add_patch(row_bg)
            ax.plot([0.045, 0.045], [y - row_h * 0.34, y + row_h * 0.34], color=color, linewidth=4.0, solid_capstyle="round")
            ax.text(0.08, y, f"#{int(row['full_rank'])}", ha="left", va="center", fontsize=10.0, fontweight="semibold", color=color)
            ax.text(0.17, y, label, ha="left", va="center", fontsize=8.6, color=text, linespacing=1.0)
            ax.text(0.54, y, full_value, ha="right", va="center", fontsize=9.4, color=text, fontweight="semibold")
            ax.text(0.70, y, subset_value, ha="right", va="center", fontsize=9.4, color=text, fontweight="semibold")
            ax.text(
                0.83,
                y,
                _fmt_delta(metric, full_mean, subset_mean),
                ha="right",
                va="center",
                color=text,
                fontsize=9.2,
            )
            ax.text(
                0.94,
                y,
                _rank_shift(float(row["full_rank"]), float(row["subset_rank"])),
                ha="center",
                va="center",
                fontsize=9.0,
                fontweight="semibold",
                color=good if np.isclose(float(row["full_rank"]), float(row["subset_rank"])) else "#b45309",
                bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#dbeafe"},
            )

        ax.text(
            0.04,
            0.07,
            "Delta = RASS-48 mean - full mean",
            ha="left",
            va="center",
            fontsize=8.4,
            color=muted,
        )

    subtitle = "Full and RASS-48 ranks are identical for every shown metric." if all_stable else "Lines show how method ranks move from full data to RASS-48."
    if full_n is not None and subset_n is not None:
        subtitle += f" Full common pool n={full_n:,}; subset n={subset_n}."

    fig.suptitle("Cross-Method Rank Stability", fontsize=15.5, fontweight="semibold", color=text, y=0.975)
    fig.text(
        0.5,
        0.915,
        subtitle,
        ha="center",
        va="top",
        color="#475569",
        fontsize=9.4,
    )
    fig.subplots_adjust(left=0.045, right=0.985, top=0.80, bottom=0.11, wspace=0.065)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON config describing subset manifest and method metric files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cross_method_ranking_fidelity"),
        help="Directory for CSV, LaTeX, figure, and markdown outputs.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config_path = args.config.resolve()
    config = _load_json(config_path)
    config_base = config_path.parent

    subset_name = str(config.get("subset_name", config_path.stem))
    subset_method = str(config.get("subset_method", "BASS"))
    subset_manifest = _resolve_path(str(config["subset_manifest"]), config_base)
    ranking_metrics = [
        str(m).strip().lower()
        for m in config.get("ranking_metrics", ["psnr"])
        if str(m).strip()
    ]
    if not ranking_metrics:
        raise ValueError("No ranking metrics configured")

    raw_method_specs = config.get("methods", [])
    if not isinstance(raw_method_specs, list) or len(raw_method_specs) < 2:
        raise ValueError("Config must provide at least two methods")

    method_specs: List[MethodSpec] = []
    for item in raw_method_specs:
        if not isinstance(item, dict):
            raise ValueError("Each method spec must be a JSON object")
        method_specs.append(
            MethodSpec(
                name=str(item["name"]),
                path=_resolve_path(str(item["path"]), config_base),
                sheet_name=(str(item["sheet_name"]) if item.get("sheet_name") else None),
            )
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "subset_name": subset_name,
                "subset_method": subset_method,
                "subset_manifest": str(subset_manifest),
                "ranking_metrics": ranking_metrics,
                "methods": [
                    {
                        "name": spec.name,
                        "path": str(spec.path),
                        "sheet_name": spec.sheet_name,
                    }
                    for spec in method_specs
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    subset_ids = _load_subset_ids(subset_manifest)
    nominal_subset_size = len(subset_ids)

    method_tables = [_canonicalize_method_table(spec) for spec in method_specs]

    coverage_rows: List[Dict[str, object]] = []
    metric_coverage_rows: List[Dict[str, object]] = []
    score_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    metric_order = {metric: idx for idx, metric in enumerate(ranking_metrics)}

    for metric in ranking_metrics:
        if metric not in METRIC_DIRECTIONS:
            raise ValueError(f"Unsupported ranking metric: {metric}")

        available_ids_per_method: Dict[str, Set[str]] = {}
        for table in method_tables:
            valid = table.df[["scene_id", metric]].dropna().copy()
            available_ids_per_method[table.name] = set(valid["scene_id"].astype(str))

        shared_full_ids = set.intersection(*available_ids_per_method.values())
        shared_subset_ids = shared_full_ids & subset_ids

        if len(shared_full_ids) == 0:
            raise RuntimeError(f"No shared full scenes for metric {metric}")
        if len(shared_subset_ids) == 0:
            raise RuntimeError(f"No shared subset scenes for metric {metric}")

        metric_coverage_rows.append(
            {
                "ranking_metric": metric,
                "shared_full_size": len(shared_full_ids),
                "shared_subset_size": len(shared_subset_ids),
                "num_methods": len(method_tables),
            }
        )

        for table in method_tables:
            coverage_rows.append(
                {
                    "ranking_metric": metric,
                    "method": table.name,
                    "path": str(table.path),
                    "full_scene_count": int(table.df["scene_id"].nunique()),
                    "nominal_subset_overlap": int(
                        table.df["scene_id"].astype(str).isin(subset_ids).sum()
                    ),
                    "shared_full_overlap": int(
                        table.df["scene_id"].astype(str).isin(shared_full_ids).sum()
                    ),
                    "shared_subset_overlap": int(
                        table.df["scene_id"].astype(str).isin(shared_subset_ids).sum()
                    ),
                }
            )

        metric_scores: List[Dict[str, object]] = []
        for table in method_tables:
            df = table.df.copy()
            df["scene_id"] = df["scene_id"].astype(str)
            full_series = pd.to_numeric(
                df[df["scene_id"].isin(shared_full_ids)][metric], errors="coerce"
            ).dropna()
            subset_series = pd.to_numeric(
                df[df["scene_id"].isin(shared_subset_ids)][metric], errors="coerce"
            ).dropna()
            if full_series.empty or subset_series.empty:
                raise RuntimeError(
                    f"{table.name}: missing {metric} values on shared scenes"
                )
            metric_scores.append(
                {
                    "ranking_metric": metric,
                    "method": table.name,
                    "full_mean": float(full_series.mean()),
                    "subset_mean": float(subset_series.mean()),
                    "full_scene_count": int(full_series.shape[0]),
                    "subset_scene_count": int(subset_series.shape[0]),
                }
            )

        metric_score_df = pd.DataFrame(metric_scores)
        higher_is_better = METRIC_DIRECTIONS[metric] == "higher"
        metric_score_df["full_rank"] = metric_score_df["full_mean"].rank(
            method="average",
            ascending=not higher_is_better,
        )
        metric_score_df["subset_rank"] = metric_score_df["subset_mean"].rank(
            method="average",
            ascending=not higher_is_better,
        )

        full_for_top = metric_score_df[["method", "full_mean"]].rename(
            columns={"full_mean": "score"}
        )
        subset_for_top = metric_score_df[["method", "subset_mean"]].rename(
            columns={"subset_mean": "score"}
        )
        full_top = _top_method_names(full_for_top, metric)
        subset_top = _top_method_names(subset_for_top, metric)

        rho = spearmanr(
            metric_score_df["full_rank"].to_numpy(dtype=float),
            metric_score_df["subset_rank"].to_numpy(dtype=float),
        ).statistic
        tau = kendalltau(
            metric_score_df["full_rank"].to_numpy(dtype=float),
            metric_score_df["subset_rank"].to_numpy(dtype=float),
        ).statistic

        pairwise_agreement = _compute_pairwise_ordering_agreement(
            metric_score_df,
            METRIC_DIRECTIONS[metric],
        )
        top_1_agreement = bool(full_top & subset_top)

        summary_rows.append(
            {
                "subset_method": subset_method,
                "ranking_metric": metric,
                "subset_size": len(shared_subset_ids),
                "spearman_rho": float(rho),
                "kendall_tau": float(tau),
                "top_1_agreement": bool(top_1_agreement),
                "pairwise_ordering_agreement": float(pairwise_agreement),
            }
        )

        score_rows.extend(metric_score_df.to_dict("records"))

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df["_metric_order"] = coverage_df["ranking_metric"].map(metric_order)
    coverage_df = coverage_df.sort_values(
        ["_metric_order", "method"], kind="mergesort"
    ).drop(columns=["_metric_order"])

    metric_coverage_df = pd.DataFrame(metric_coverage_rows)
    metric_coverage_df["_metric_order"] = metric_coverage_df["ranking_metric"].map(metric_order)
    metric_coverage_df = metric_coverage_df.sort_values(
        ["_metric_order"], kind="mergesort"
    ).drop(columns=["_metric_order"])

    scores_df = pd.DataFrame(score_rows)
    scores_df["_metric_order"] = scores_df["ranking_metric"].map(metric_order)
    scores_df = scores_df.sort_values(
        ["_metric_order", "full_rank", "method"], kind="mergesort"
    ).drop(columns=["_metric_order"])

    summary_df = pd.DataFrame(summary_rows)
    summary_df["_metric_order"] = summary_df["ranking_metric"].map(metric_order)
    summary_df = summary_df.sort_values(
        ["_metric_order"], kind="mergesort"
    ).drop(columns=["_metric_order"])

    coverage_df.to_csv(output_dir / "coverage_summary.csv", index=False)
    metric_coverage_df.to_csv(output_dir / "metric_coverage_summary.csv", index=False)
    scores_df.to_csv(output_dir / "method_scores_by_metric.csv", index=False)
    summary_df.to_csv(output_dir / "ranking_fidelity_summary.csv", index=False)
    _write_latex_table(output_dir / "ranking_fidelity_summary.tex", summary_df)
    _write_interpretation(
        output_dir / "interpretation.md",
        subset_name=subset_name,
        subset_method=subset_method,
        subset_manifest=subset_manifest,
        nominal_subset_size=nominal_subset_size,
        metric_coverage_df=metric_coverage_df,
        summary_df=summary_df,
    )
    _make_rank_figure(
        output_dir / "rank_stability.png",
        scores_df,
        metrics=ranking_metrics,
    )

    LOGGER.info("Wrote outputs to %s", output_dir)
    LOGGER.info(
        "Nominal subset size=%d | metrics=%s",
        nominal_subset_size,
        ",".join(ranking_metrics),
    )


if __name__ == "__main__":
    main()
