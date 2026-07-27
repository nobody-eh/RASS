#!/usr/bin/env python3
"""Analyze regime-level fidelity of exported benchmark subsets.

This diagnostic reuses an existing descriptor-space regime assignment and
compares full-dataset regime statistics against the regime statistics induced
by exported subset manifests. It reports mean regime-level metric gaps and a
descriptive KS distance within each regime.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


LOGGER = logging.getLogger("regime_fidelity")

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "PSNR": 0.5,
    "SSIM": 0.01,
    "LPIPS": 0.01,
    "KS": 0.14,
}

DEFAULT_HOLDOUT_REPORT = Path("sweep_cluster_k/holdout_protocol_v3/holdout_protocol_report.json")
DEFAULT_METRICS_PATH = Path("sweep_cluster_k/holdout_protocol_v2/joint_selection/zipnerf.xlsx")


@dataclass(frozen=True)
class SubsetSpec:
    label: str
    path: Path
    source: str


def _normalize_col_name(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(col).lower())


def _pick_first(norm_to_orig: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
    for cand in candidates:
        if cand in norm_to_orig:
            return norm_to_orig[cand]
    return None


def _resolve_path(raw: str, base_dir: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_table(path: Path, sheet_name: Optional[str]) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        if sheet_name is not None:
            return pd.read_excel(path, sheet_name=sheet_name)
        workbook = pd.ExcelFile(path)
        if not workbook.sheet_names:
            raise ValueError(f"No sheets found in workbook: {path}")
        return workbook.parse(workbook.sheet_names[0])
    raise ValueError(f"Unsupported table type: {path}")


def _canonicalize_metric_table(path: Path, sheet_name: Optional[str]) -> pd.DataFrame:
    raw = _read_table(path, sheet_name)
    if raw.empty:
        raise ValueError(f"Metric table is empty: {path}")

    norm_to_orig: Dict[str, str] = {}
    for col in raw.columns:
        norm = _normalize_col_name(col)
        if norm not in norm_to_orig:
            norm_to_orig[norm] = str(col)

    id_col = _pick_first(norm_to_orig, ("dishid", "experimentname", "sceneid", "id"))
    if id_col is None:
        raise ValueError(f"Unable to find scene identifier column in {path}")

    metric_map = {
        "PSNR": _pick_first(
            norm_to_orig,
            ("psnravgfrommse", "psnravgmse", "psnr", "pnsr", "psnrmean"),
        ),
        "SSIM": _pick_first(norm_to_orig, ("ssim", "ssimmean")),
        "LPIPS": _pick_first(norm_to_orig, ("lpips", "lpipsmean")),
    }

    out = pd.DataFrame()
    out["dish_id"] = raw[id_col].astype(str).str.strip()
    out = out[out["dish_id"] != ""].copy()
    for metric_name, src_col in metric_map.items():
        if src_col is None:
            out[metric_name] = np.nan
        else:
            out[metric_name] = pd.to_numeric(raw[src_col], errors="coerce")

    out = (
        out.groupby("dish_id", as_index=False)[["PSNR", "SSIM", "LPIPS"]]
        .mean(numeric_only=True)
        .reset_index(drop=True)
    )
    all_missing = out[["PSNR", "SSIM", "LPIPS"]].isna().all(axis=1)
    out = out[~all_missing].reset_index(drop=True)
    return out


def _load_mapping_csv(path: Path) -> pd.DataFrame:
    mapping = pd.read_csv(path)
    norm_to_orig = {_normalize_col_name(c): c for c in mapping.columns}
    id_col = _pick_first(norm_to_orig, ("dishid", "sceneid", "id"))
    cluster_col = _pick_first(norm_to_orig, ("cluster", "regime"))
    if id_col is None or cluster_col is None:
        raise ValueError(f"{path}: expected dish id and cluster columns")

    out = pd.DataFrame()
    out["dish_id"] = mapping[id_col].astype(str).str.strip()
    out["cluster"] = pd.to_numeric(mapping[cluster_col], errors="coerce")
    out = out.dropna(subset=["dish_id", "cluster"]).copy()
    out["cluster"] = out["cluster"].astype(int)
    out = out.drop_duplicates(subset=["dish_id"], keep="first").reset_index(drop=True)
    return out


def _load_subset_ids(path: Path) -> Tuple[set[str], int]:
    df = pd.read_csv(path)
    norm_to_orig = {_normalize_col_name(c): c for c in df.columns}
    id_col = _pick_first(norm_to_orig, ("dishid", "experimentname", "sceneid", "id"))
    if id_col is None:
        raise ValueError(f"{path}: unable to find subset identifier column")
    ids = set(df[id_col].astype(str).str.strip())
    ids.discard("")
    return ids, int(len(df))


def _latex_escape(text: str) -> str:
    out = str(text)
    out = out.replace("\\", "\\textbackslash{}")
    out = out.replace("_", "\\_")
    out = out.replace("%", "\\%")
    out = out.replace("&", "\\&")
    out = out.replace("#", "\\#")
    return out


def _find_mapping_csv_for_k(repo_root: Path, k: int) -> Path:
    run_dir = repo_root / "sweep_cluster_k" / f"k_{int(k)}"
    matches = sorted(run_dir.glob("*_dish_cluster_mapping.csv"))
    if not matches:
        raise FileNotFoundError(f"No dish_cluster_mapping CSV found in {run_dir}")
    return matches[0].resolve()


def _discover_from_holdout_report(
    holdout_report_path: Path,
    repo_root: Path,
) -> Tuple[Path, List[SubsetSpec]]:
    report = _load_json(holdout_report_path)

    cluster_mapping_csv: Optional[Path] = None
    tune_validation = report.get("tune_validation", {})
    if isinstance(tune_validation, dict) and tune_validation.get("cluster_mapping_csv"):
        cluster_mapping_csv = _resolve_path(
            str(tune_validation["cluster_mapping_csv"]),
            repo_root,
        )
    if cluster_mapping_csv is None:
        joint_rec = report.get("joint_selection_recommendation", {})
        if not isinstance(joint_rec, dict) or "recommended_k" not in joint_rec:
            raise ValueError(
                f"{holdout_report_path}: unable to infer cluster mapping from holdout report"
            )
        cluster_mapping_csv = _find_mapping_csv_for_k(repo_root, int(joint_rec["recommended_k"]))

    subset_specs: List[SubsetSpec] = []
    joint_rec = report.get("joint_selection_recommendation", {})
    if isinstance(joint_rec, dict) and joint_rec.get("exported_manifest"):
        subset_specs.append(
            SubsetSpec(
                label="BASS-joint",
                path=_resolve_path(str(joint_rec["exported_manifest"]), repo_root),
                source="holdout_protocol_v3:joint_selection",
            )
        )

    split_recs = report.get("split_budget_recommendations", {})
    if isinstance(split_recs, dict):
        for split_name in ("tune", "test"):
            split_rec = split_recs.get(split_name, {})
            if isinstance(split_rec, dict) and split_rec.get("exported_manifest"):
                subset_specs.append(
                    SubsetSpec(
                        label=f"BASS-{split_name}",
                        path=_resolve_path(
                            str(split_rec["exported_manifest"]),
                            repo_root,
                        ),
                        source=f"holdout_protocol_v3:{split_name}_budget_eval",
                    )
                )

    if not subset_specs:
        raise ValueError(f"{holdout_report_path}: no exported subset manifests found")
    return cluster_mapping_csv.resolve(), subset_specs


def _parse_subset_arg(raw: str, repo_root: Path) -> SubsetSpec:
    if "=" not in str(raw):
        raise ValueError("--subset must be provided as label=path")
    label, path_str = str(raw).split("=", 1)
    label = label.strip()
    path_str = path_str.strip()
    if not label or not path_str:
        raise ValueError("--subset must be provided as label=path")
    return SubsetSpec(
        label=label,
        path=_resolve_path(path_str, repo_root),
        source="cli",
    )


def _discover_missing_baselines(repo_root: Path) -> List[str]:
    baseline_summary = repo_root / "sweep_cluster_k" / "baseline_comparison_lpips_ks" / "baseline_min_size_summary.csv"
    if not baseline_summary.exists():
        return []

    df = pd.read_csv(baseline_summary)
    if "baseline" not in df.columns:
        return []

    missing: List[str] = []
    for baseline_name in ("random_uniform", "facility_location"):
        if baseline_name not in set(df["baseline"].astype(str)):
            continue
        matches: List[Path] = []
        for suffix in ("*.csv", "*.json"):
            for path in repo_root.rglob(suffix):
                name = path.name.lower()
                if baseline_name in name and ("subset" in name or "manifest" in name):
                    matches.append(path)
        if not matches:
            missing.append(baseline_name)
    return missing


def _merge_metrics_with_mapping(metrics_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    merged = metrics_df.merge(mapping_df[["dish_id", "cluster"]], on="dish_id", how="inner")
    merged["cluster"] = pd.to_numeric(merged["cluster"], errors="coerce")
    merged = merged.dropna(subset=["cluster"]).copy()
    merged["cluster"] = merged["cluster"].astype(int)
    return merged.reset_index(drop=True)


def _metric_gap_and_ks(full_vals: np.ndarray, subset_vals: np.ndarray) -> Tuple[float, float]:
    full_mean = float(np.mean(full_vals)) if full_vals.size > 0 else float("nan")
    subset_mean = float(np.mean(subset_vals)) if subset_vals.size > 0 else float("nan")
    gap = subset_mean - full_mean
    if full_vals.size >= 2 and subset_vals.size >= 2:
        ks_val = float(ks_2samp(full_vals, subset_vals).statistic)
    else:
        ks_val = float("nan")
    return float(gap), ks_val


def _compute_regime_breakdown(
    full_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    subset_label: str,
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    regimes = sorted(int(v) for v in full_df["cluster"].dropna().unique())
    for regime in regimes:
        full_regime = full_df[full_df["cluster"] == regime].copy()
        subset_regime = subset_df[subset_df["cluster"] == regime].copy()
        row: Dict[str, object] = {
            "subset_method": subset_label,
            "regime": int(regime),
            "full_count": int(full_regime.shape[0]),
            "subset_count": int(subset_regime.shape[0]),
        }

        normalized_components: List[float] = []
        max_ks = float("nan")
        within_tolerance = True

        for metric in ("PSNR", "SSIM", "LPIPS"):
            full_vals = pd.to_numeric(full_regime[metric], errors="coerce").dropna().to_numpy(dtype=float)
            subset_vals = pd.to_numeric(subset_regime[metric], errors="coerce").dropna().to_numpy(dtype=float)
            gap, ks_val = _metric_gap_and_ks(full_vals, subset_vals)
            abs_gap = float(abs(gap)) if not math.isnan(gap) else float("nan")

            row[f"{metric.lower()}_gap"] = gap
            row[f"abs_{metric.lower()}_gap"] = abs_gap
            row[f"{metric.lower()}_ks"] = ks_val

            if not math.isnan(abs_gap):
                normalized_components.append(abs_gap / float(thresholds[metric]))
                if abs_gap > float(thresholds[metric]):
                    within_tolerance = False
            else:
                within_tolerance = False

            if not math.isnan(ks_val):
                if math.isnan(max_ks):
                    max_ks = ks_val
                else:
                    max_ks = max(max_ks, ks_val)
                normalized_components.append(ks_val / float(thresholds["KS"]))
                if ks_val > float(thresholds["KS"]):
                    within_tolerance = False

        row["max_ks_distance"] = max_ks
        row["within_tolerance"] = bool(within_tolerance)
        row["normalized_discrepancy"] = (
            float(max(normalized_components)) if normalized_components else float("nan")
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("regime", kind="mergesort").reset_index(drop=True)


def _summarize_subset(
    subset_label: str,
    subset_source: str,
    subset_manifest: Path,
    nominal_subset_size: int,
    evaluated_subset_size: int,
    per_regime_df: pd.DataFrame,
) -> Dict[str, object]:
    if per_regime_df.empty:
        raise ValueError(f"{subset_label}: no regime rows were computed")

    worst_idx = per_regime_df["normalized_discrepancy"].astype(float).idxmax()
    worst_row = per_regime_df.loc[worst_idx]
    return {
        "subset_method": subset_label,
        "subset_source": subset_source,
        "subset_manifest": str(subset_manifest),
        "subset_size": int(nominal_subset_size),
        "evaluated_subset_size": int(evaluated_subset_size),
        "num_regimes": int(per_regime_df["regime"].nunique()),
        "mean_abs_regime_psnr_gap": float(per_regime_df["abs_psnr_gap"].mean()),
        "mean_abs_regime_ssim_gap": float(per_regime_df["abs_ssim_gap"].mean()),
        "mean_abs_regime_lpips_gap": float(per_regime_df["abs_lpips_gap"].mean()),
        "number_of_regimes_within_tolerance": int(per_regime_df["within_tolerance"].sum()),
        "worst_regime_discrepancy": float(worst_row["normalized_discrepancy"]),
        "worst_regime": int(worst_row["regime"]),
        "mean_regime_max_ks": float(per_regime_df["max_ks_distance"].dropna().mean()),
    }


def _write_latex_table(path: Path, summary_df: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\hline",
        "Subset & Size & Mean $|\\Delta|$ PSNR & Mean $|\\Delta|$ SSIM & Mean $|\\Delta|$ LPIPS & Regimes in tol. & Worst disc. \\\\",
        "\\hline",
    ]
    for row in summary_df.to_dict("records"):
        lines.append(
            "{} & {} & {:.3f} & {:.4f} & {:.4f} & {} & {:.2f} \\\\".format(
                _latex_escape(row["subset_method"]),
                int(row["subset_size"]),
                float(row["mean_abs_regime_psnr_gap"]),
                float(row["mean_abs_regime_ssim_gap"]),
                float(row["mean_abs_regime_lpips_gap"]),
                int(row["number_of_regimes_within_tolerance"]),
                float(row["worst_regime_discrepancy"]),
            )
        )
    lines += ["\\hline", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_figure(
    per_regime_df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    method_order: Sequence[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        from matplotlib.patches import Rectangle
    except Exception as exc:  # pragma: no cover - optional dependency behavior
        LOGGER.warning("Skipping figure export because matplotlib is unavailable: %s", exc)
        return

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
        }
    )

    pivot = per_regime_df.pivot(index="subset_method", columns="regime", values="normalized_discrepancy")
    pivot = pivot.reindex(list(method_order))
    pivot = pivot.sort_index(axis=1)
    display_index = [str(label).replace("BASS", "RASS") for label in pivot.index.tolist()]

    vals = pivot.to_numpy(dtype=float)
    vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
    vlim = max(18.0, math.ceil(vmax))
    cmap = LinearSegmentedColormap.from_list(
        "regime_fidelity",
        [
            (0.00, "#f8fafc"),
            (0.07, "#d1fae5"),
            (0.18, "#fde68a"),
            (0.50, "#f97316"),
            (0.78, "#dc2626"),
            (1.00, "#7f1d1d"),
        ],
    )
    norm = Normalize(vmin=0.0, vmax=vlim)

    fig = plt.figure(figsize=(10.1, 4.7))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.12, 1.0],
        width_ratios=[1.0, 0.23],
        hspace=0.16,
        wspace=0.055,
    )
    cax = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    ax_side = fig.add_subplot(gs[1, 1])

    im = ax.imshow(vals, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(pivot.shape[1]), labels=[str(v) for v in pivot.columns.tolist()])
    ax.set_yticks(np.arange(pivot.shape[0]), labels=display_index)
    ax.set_xlabel("Regime")
    ax.set_ylabel("")
    ax.tick_params(axis="both", length=0, colors="#334155")
    ax.set_xticks(np.arange(-0.5, pivot.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, pivot.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_facecolor("#ffffff")

    for i in range(pivot.shape[0]):
        row_vals = vals[i, :]
        finite = np.isfinite(row_vals)
        if finite.any():
            worst_j = int(np.nanargmax(row_vals))
            ax.add_patch(
                Rectangle(
                    (worst_j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="#0f172a",
                    linewidth=2.2,
                    zorder=4,
                )
            )
        for j in range(pivot.shape[1]):
            val = vals[i, j]
            if math.isnan(float(val)):
                txt = "-"
                color = "#334155"
            else:
                txt = f"{float(val):.1f}"
                color = "white" if float(val) >= 8.0 else "#0f172a"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8.8, fontweight="semibold")

    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Normalized discrepancy  (1.0 = tolerance boundary; lower is better)", color="#475569", labelpad=7)
    cbar.set_ticks([0, 1, 5, 10, 15, vlim])
    cbar.ax.tick_params(labelsize=8.5, colors="#475569", length=0)
    cbar.outline.set_visible(False)
    cax.axvline(1.0, color="#047857", linewidth=2.0, ymin=0.15, ymax=0.85)
    cax.text(1.0, 1.22, "target", transform=cax.get_xaxis_transform(), ha="center", va="bottom", color="#047857", fontsize=8.2, fontweight="semibold")

    ax_side.set_axis_off()
    ax_side.set_xlim(0, 1)
    ax_side.set_ylim(0, 1)
    ax_side.text(0.0, 0.98, "Row Summary", ha="left", va="top", fontsize=11.5, fontweight="semibold", color="#0f172a")
    ax_side.text(
        0.0,
        0.89,
        "Outlined cells mark the worst regime for each subset.",
        ha="left",
        va="top",
        fontsize=8.4,
        color="#64748b",
        wrap=True,
    )

    cards_y = [0.68, 0.45, 0.22]
    for idx, label in enumerate(pivot.index.tolist()):
        sub = per_regime_df[per_regime_df["subset_method"] == label].copy()
        if sub.empty:
            continue
        worst_row = sub.loc[sub["normalized_discrepancy"].idxmax()]
        mean_val = float(sub["normalized_discrepancy"].mean())
        worst_val = float(worst_row["normalized_discrepancy"])
        worst_regime = int(worst_row["regime"])
        y = cards_y[idx] if idx < len(cards_y) else 0.08
        ax_side.add_patch(
            Rectangle(
                (0.0, y - 0.12),
                0.98,
                0.17,
                transform=ax_side.transAxes,
                facecolor="#f8fafc",
                edgecolor="#e2e8f0",
                linewidth=1.0,
                zorder=-1,
            )
        )
        ax_side.text(0.05, y + 0.02, str(label).replace("BASS", "RASS"), ha="left", va="center", fontsize=9.2, fontweight="semibold", color="#0f172a")
        ax_side.text(0.05, y - 0.045, f"worst R{worst_regime}: {worst_val:.1f}x", ha="left", va="center", fontsize=8.6, color="#b91c1c", fontweight="semibold")
        ax_side.text(0.05, y - 0.095, f"mean: {mean_val:.1f}x", ha="left", va="center", fontsize=8.4, color="#475569")

    all_over_target = bool(np.nanmin(vals) > 1.0) if np.isfinite(vals).any() else False
    subtitle = "Every cell is above the 1.0 tolerance boundary; darker cells indicate larger regime-level mismatch."
    if not all_over_target:
        subtitle = "Cells at or below 1.0 meet the tolerance; darker cells indicate larger regime-level mismatch."

    fig.suptitle("Regime-Level Fidelity Across RASS-48 Variants", fontsize=15.0, fontweight="semibold", color="#0f172a", y=0.985)
    fig.text(0.5, 0.925, subtitle, ha="center", va="top", color="#475569", fontsize=9.3)
    fig.subplots_adjust(left=0.09, right=0.975, top=0.82, bottom=0.15)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_markdown_note(
    path: Path,
    cluster_mapping_csv: Path,
    metrics_path: Path,
    full_scene_count: int,
    num_regimes: int,
    thresholds: Dict[str, float],
    summary_df: pd.DataFrame,
    missing_baselines: Sequence[str],
    run_command: str,
) -> None:
    lines = [
        "# Regime-Level Fidelity",
        "",
        "## Inputs",
        f"- Regime assignments: `{cluster_mapping_csv}`",
        f"- Metric table: `{metrics_path}`",
        f"- Full metric overlap: **{full_scene_count}** scenes across **{num_regimes}** regimes",
        "",
        "## Command",
        "```bash",
        run_command,
        "```",
        "",
        "## Outputs",
        "- `regime_fidelity_summary.csv`",
        "- `regime_fidelity_summary.tex`",
        "- `regime_fidelity_by_regime.csv`",
        "- `regime_fidelity_heatmap.pdf`",
        "- `regime_fidelity_heatmap.png`",
        "- `run_config.json`",
        "",
        "## How To Read It",
        f"- Mean regime-level gaps are subset regime means minus full-dataset regime means, aggregated as absolute values across regimes.",
        f"- `number_of_regimes_within_tolerance` counts regimes where `|PSNR gap| <= {thresholds['PSNR']}`, `|SSIM gap| <= {thresholds['SSIM']}`, `|LPIPS gap| <= {thresholds['LPIPS']}`, and the largest per-regime KS distance stays below `{thresholds['KS']}`.",
        "- `worst_regime_discrepancy` is the largest normalized discrepancy over all regimes and metrics, where `1.0` means the threshold boundary and larger values indicate larger deviations.",
        "- KS distances here are descriptive diagnostics, not hypothesis tests; with only a few scenes per regime in a 48-scene subset they mainly indicate coarse shape mismatch.",
        "",
        "## Availability",
    ]

    if missing_baselines:
        pretty = ", ".join(f"`{name}`" for name in missing_baselines)
        lines.append(
            f"- Baseline summary CSVs exist for {pretty}, but no matching subset manifests were found in the repo, so those baselines are not included in this regime-level analysis."
        )
    else:
        lines.append("- No missing baseline manifests were detected by the built-in search.")

    lines += [
        "",
        "## Result Snapshot",
        "",
        "| Subset | Size | Mean abs PSNR gap | Mean abs SSIM gap | Mean abs LPIPS gap | Regimes in tol. | Worst regime | Worst disc. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary_df.to_dict("records"):
        lines.append(
            f"| {row['subset_method']} | {int(row['subset_size'])} | "
            f"{float(row['mean_abs_regime_psnr_gap']):.3f} | "
            f"{float(row['mean_abs_regime_ssim_gap']):.4f} | "
            f"{float(row['mean_abs_regime_lpips_gap']):.4f} | "
            f"{int(row['number_of_regimes_within_tolerance'])} | "
            f"{int(row['worst_regime'])} | "
            f"{float(row['worst_regime_discrepancy']):.2f} |"
        )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout-report",
        type=Path,
        default=DEFAULT_HOLDOUT_REPORT,
        help="Holdout protocol report JSON used to auto-discover exported BASS subsets.",
    )
    parser.add_argument(
        "--cluster-mapping-csv",
        type=Path,
        default=None,
        help="Optional override for the regime assignment CSV.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Per-scene metric table (.csv/.xlsx) used for regime statistics.",
    )
    parser.add_argument(
        "--metrics-sheet-name",
        type=str,
        default=None,
        help="Optional sheet name for Excel metric files.",
    )
    parser.add_argument(
        "--subset",
        action="append",
        default=[],
        help="Additional subset manifest in label=path form. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/regime_fidelity/holdout_protocol_v3_zipnerf"),
        help="Directory for the exported analysis artifacts.",
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

    repo_root = Path(__file__).resolve().parents[1]
    holdout_report = _resolve_path(str(args.holdout_report), repo_root)
    metrics_path = _resolve_path(str(args.metrics_path), repo_root)
    output_dir = _resolve_path(str(args.output_dir), repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cluster_mapping_csv is not None:
        cluster_mapping_csv = _resolve_path(str(args.cluster_mapping_csv), repo_root)
        subset_specs: List[SubsetSpec] = []
    else:
        cluster_mapping_csv, subset_specs = _discover_from_holdout_report(holdout_report, repo_root)

    extra_subset_specs = [_parse_subset_arg(raw, repo_root) for raw in args.subset]
    subset_specs.extend(extra_subset_specs)
    if not subset_specs:
        raise ValueError("No subset manifests were provided or discovered")

    deduped_specs: List[SubsetSpec] = []
    seen_labels: set[str] = set()
    for spec in subset_specs:
        if spec.label in seen_labels:
            raise ValueError(f"Duplicate subset label detected: {spec.label}")
        seen_labels.add(spec.label)
        deduped_specs.append(spec)
    subset_specs = deduped_specs

    thresholds = dict(DEFAULT_THRESHOLDS)
    metrics_df = _canonicalize_metric_table(metrics_path, args.metrics_sheet_name)
    mapping_df = _load_mapping_csv(cluster_mapping_csv)
    full_df = _merge_metrics_with_mapping(metrics_df, mapping_df)
    if full_df.empty:
        raise ValueError("Metric table and regime mapping do not overlap")

    summary_rows: List[Dict[str, object]] = []
    per_regime_tables: List[pd.DataFrame] = []
    missing_subset_rows: List[Dict[str, object]] = []

    for spec in subset_specs:
        if not spec.path.exists():
            missing_subset_rows.append(
                {
                    "subset_method": spec.label,
                    "subset_source": spec.source,
                    "subset_manifest": str(spec.path),
                    "status": "missing_manifest",
                }
            )
            LOGGER.warning("Skipping missing subset manifest: %s", spec.path)
            continue

        subset_ids, nominal_subset_size = _load_subset_ids(spec.path)
        subset_df = full_df[full_df["dish_id"].isin(subset_ids)].copy()
        if subset_df.empty:
            missing_subset_rows.append(
                {
                    "subset_method": spec.label,
                    "subset_source": spec.source,
                    "subset_manifest": str(spec.path),
                    "status": "no_metric_overlap",
                }
            )
            LOGGER.warning("Skipping subset with no metric overlap: %s", spec.path)
            continue

        per_regime_df = _compute_regime_breakdown(
            full_df=full_df,
            subset_df=subset_df,
            subset_label=spec.label,
            thresholds=thresholds,
        )
        per_regime_tables.append(per_regime_df)
        summary_rows.append(
            _summarize_subset(
                subset_label=spec.label,
                subset_source=spec.source,
                subset_manifest=spec.path,
                nominal_subset_size=nominal_subset_size,
                evaluated_subset_size=int(subset_df["dish_id"].nunique()),
                per_regime_df=per_regime_df,
            )
        )

    if not summary_rows:
        raise RuntimeError("No subset manifests could be evaluated")

    summary_df = pd.DataFrame(summary_rows)
    method_order = [
        spec.label
        for spec in subset_specs
        if spec.label in set(summary_df["subset_method"].astype(str))
    ]
    order_map = {label: idx for idx, label in enumerate(method_order)}
    summary_df["subset_order"] = summary_df["subset_method"].map(order_map).fillna(len(order_map))
    summary_df = summary_df.sort_values(
        ["subset_size", "subset_order", "subset_method"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    summary_df = summary_df.drop(columns=["subset_order"])
    per_regime_df = pd.concat(per_regime_tables, ignore_index=True)
    method_order = summary_df["subset_method"].astype(str).tolist()

    summary_path = output_dir / "regime_fidelity_summary.csv"
    per_regime_path = output_dir / "regime_fidelity_by_regime.csv"
    latex_path = output_dir / "regime_fidelity_summary.tex"
    readme_path = output_dir / "README.md"
    run_config_path = output_dir / "run_config.json"
    figure_pdf = output_dir / "regime_fidelity_heatmap.pdf"
    figure_png = output_dir / "regime_fidelity_heatmap.png"

    summary_df.to_csv(summary_path, index=False)
    per_regime_df.to_csv(per_regime_path, index=False)
    _write_latex_table(latex_path, summary_df)
    _save_figure(per_regime_df, figure_pdf, figure_png, method_order=method_order)

    missing_baselines = _discover_missing_baselines(repo_root)
    run_command = (
        "python3 scripts/analyze_regime_fidelity.py "
        f"--holdout-report {holdout_report.relative_to(repo_root)} "
        f"--metrics-path {metrics_path.relative_to(repo_root)} "
        f"--output-dir {output_dir.relative_to(repo_root)}"
    )

    _write_markdown_note(
        path=readme_path,
        cluster_mapping_csv=cluster_mapping_csv,
        metrics_path=metrics_path,
        full_scene_count=int(full_df["dish_id"].nunique()),
        num_regimes=int(full_df["cluster"].nunique()),
        thresholds=thresholds,
        summary_df=summary_df,
        missing_baselines=missing_baselines,
        run_command=run_command,
    )

    run_payload = {
        "holdout_report": str(holdout_report),
        "cluster_mapping_csv": str(cluster_mapping_csv),
        "metrics_path": str(metrics_path),
        "metrics_sheet_name": args.metrics_sheet_name,
        "thresholds": thresholds,
        "full_scene_count": int(full_df["dish_id"].nunique()),
        "num_regimes": int(full_df["cluster"].nunique()),
        "subset_specs": [
            {
                "label": spec.label,
                "path": str(spec.path),
                "source": spec.source,
            }
            for spec in subset_specs
        ],
        "missing_subset_rows": missing_subset_rows,
        "missing_baseline_manifests": missing_baselines,
        "outputs": {
            "summary_csv": str(summary_path),
            "summary_tex": str(latex_path),
            "per_regime_csv": str(per_regime_path),
            "figure_pdf": str(figure_pdf),
            "figure_png": str(figure_png),
            "readme_md": str(readme_path),
        },
    }
    run_config_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
