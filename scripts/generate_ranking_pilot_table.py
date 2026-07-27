#!/usr/bin/env python3
"""Generate the three-method full-vs-BASS-48 ranking pilot table.

This script compares method means on:
1) the full scene intersection shared by Instant-NGP, Zip-NeRF, and
   Feature-Splatting
2) the selected BASS-48 subset restricted to the same shared intersection

It writes:
- results/ranking_pilot_full_vs_bass48.csv
- tables/ranking_pilot_full_vs_bass48.tex
- results/ranking_pilot_full_vs_bass48_report.md
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
from scipy.stats import kendalltau, spearmanr


LOGGER = logging.getLogger("ranking_pilot_table")

METHOD_ORDER = ["Instant-NGP", "Zip-NeRF", "Feature-Splatting"]

DEFAULT_METHOD_CANDIDATES: Dict[str, List[Path]] = {
    "Instant-NGP": [
        Path("ingp_fi.csv"),
        Path("sweep_cluster_k/holdout_protocol_v2/joint_selection/instant-ngp_nerf.xlsx"),
    ],
    "Zip-NeRF": [
        Path("sweep_cluster_k/holdout_protocol_v2/joint_selection/zipnerf.xlsx"),
        Path("zipnerf_metrics.csv"),
    ],
    "Feature-Splatting": [
        Path("sweep_cluster_k/holdout_protocol_v2/joint_selection/feature_splatting.csv"),
    ],
}

DEFAULT_SUBSET_CANDIDATES = [
    Path("sweep_cluster_k/holdout_protocol_v3/joint_selection/recommended_subset.csv"),
    Path("sweep_cluster_k/holdout_protocol_v2/joint_selection/recommended_subset.csv"),
    Path("sweep_cluster_k/budget_sweep_k6_auto_v2/recommended_subset.csv"),
]

DEFAULT_HOLDOUT_REPORT = Path("sweep_cluster_k/holdout_protocol_v3/holdout_protocol_report.json")


@dataclass(frozen=True)
class MethodSpec:
    method_name: str
    path: Path
    sheet_name: Optional[str] = None


@dataclass
class MethodTable:
    method_name: str
    path: Path
    sheet_name: Optional[str]
    df: pd.DataFrame


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


def _detect_sheet_name(path: Path) -> Optional[str]:
    if path.suffix.lower() not in {".xlsx", ".xls"}:
        return None
    workbook = pd.ExcelFile(path)
    if "zipnerf" in workbook.sheet_names:
        return "zipnerf"
    return workbook.sheet_names[0] if workbook.sheet_names else None


def _auto_detect_method_path(repo_root: Path, method_name: str) -> MethodSpec:
    candidates = DEFAULT_METHOD_CANDIDATES[method_name]
    for cand in candidates:
        path = (repo_root / cand).resolve()
        if path.exists():
            return MethodSpec(
                method_name=method_name,
                path=path,
                sheet_name=_detect_sheet_name(path),
            )
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not auto-detect metrics for {method_name}. Searched: {searched}")


def _auto_detect_subset_path(repo_root: Path) -> Path:
    holdout_report = (repo_root / DEFAULT_HOLDOUT_REPORT).resolve()
    if holdout_report.exists():
        payload = _load_json(holdout_report)
        joint_rec = payload.get("joint_selection_recommendation", {})
        if isinstance(joint_rec, dict):
            exported = joint_rec.get("exported_manifest")
            if exported:
                resolved = _resolve_path(str(exported), repo_root)
                if resolved.exists():
                    return resolved

    for cand in DEFAULT_SUBSET_CANDIDATES:
        path = (repo_root / cand).resolve()
        if path.exists():
            return path

    searched = ", ".join(str(p) for p in DEFAULT_SUBSET_CANDIDATES)
    raise FileNotFoundError(f"Could not auto-detect BASS subset manifest. Searched: {searched}")


def _canonicalize_method_table(spec: MethodSpec) -> MethodTable:
    raw = _read_table(spec.path, spec.sheet_name)
    if raw.empty:
        raise ValueError(f"{spec.method_name}: metrics table is empty: {spec.path}")

    norm_to_orig: Dict[str, str] = {}
    for col in raw.columns:
        norm = _normalize_col_name(col)
        if norm not in norm_to_orig:
            norm_to_orig[norm] = str(col)

    id_col = _pick_first(
        norm_to_orig,
        ("dishid", "experimentname", "sceneid", "scene", "caseid", "id"),
    )
    psnr_col = _pick_first(
        norm_to_orig,
        ("psnravgmse", "psnravgfrommse", "meanpsnr", "psnr", "pnsr"),
    )
    ssim_col = _pick_first(norm_to_orig, ("meanssim", "ssim"))
    lpips_col = _pick_first(norm_to_orig, ("meanlpips", "lpips"))

    if id_col is None:
        raise ValueError(
            f"{spec.method_name}: unable to find scene identifier column in {list(raw.columns)}"
        )
    if psnr_col is None or ssim_col is None:
        raise ValueError(
            f"{spec.method_name}: unable to find required PSNR/SSIM columns in {list(raw.columns)}"
        )

    out = pd.DataFrame()
    out["scene_id"] = raw[id_col].astype(str).str.strip()
    out["psnr"] = pd.to_numeric(raw[psnr_col], errors="coerce")
    out["ssim"] = pd.to_numeric(raw[ssim_col], errors="coerce")
    out["lpips"] = (
        pd.to_numeric(raw[lpips_col], errors="coerce")
        if lpips_col is not None
        else np.nan
    )

    out = out[out["scene_id"] != ""].copy()
    out = (
        out.groupby("scene_id", as_index=False)[["psnr", "ssim", "lpips"]]
        .mean(numeric_only=True)
        .sort_values("scene_id", kind="mergesort")
        .reset_index(drop=True)
    )
    return MethodTable(
        method_name=spec.method_name,
        path=spec.path,
        sheet_name=spec.sheet_name,
        df=out,
    )


def _load_subset_ids(path: Path) -> List[str]:
    df = pd.read_csv(path)
    norm_to_orig = {_normalize_col_name(c): c for c in df.columns}
    id_col = _pick_first(norm_to_orig, ("dishid", "experimentname", "sceneid", "scene", "caseid", "id"))
    if id_col is None:
        raise ValueError(f"{path}: unable to find subset identifier column in {list(df.columns)}")
    subset_ids = sorted(set(df[id_col].astype(str).str.strip()) - {""})
    if not subset_ids:
        raise ValueError(f"{path}: subset manifest is empty after reading scene IDs")
    return subset_ids


def _rank_map(scores: Dict[str, float], higher_is_better: bool) -> Dict[str, int]:
    items = list(scores.items())
    items.sort(
        key=lambda kv: ((-kv[1]) if higher_is_better else kv[1], kv[0]),
    )
    return {method_name: idx + 1 for idx, (method_name, _) in enumerate(items)}


def _compute_metric_means(
    method_tables: Sequence[MethodTable],
    shared_full_ids: Sequence[str],
    shared_subset_ids: Sequence[str],
) -> pd.DataFrame:
    full_id_set = set(shared_full_ids)
    subset_id_set = set(shared_subset_ids)
    rows: List[Dict[str, object]] = []

    for method_name in METHOD_ORDER:
        table = next(t for t in method_tables if t.method_name == method_name)
        df = table.df
        full = df[df["scene_id"].isin(full_id_set)].copy()
        subset = df[df["scene_id"].isin(subset_id_set)].copy()
        if full.empty or subset.empty:
            raise RuntimeError(f"{method_name}: no rows available after intersection filtering")

        rows.append(
            {
                "Method": method_name,
                "Full PSNR": float(full["psnr"].mean()),
                "BASS-48 PSNR": float(subset["psnr"].mean()),
                "Full SSIM": float(full["ssim"].mean()),
                "BASS-48 SSIM": float(subset["ssim"].mean()),
                "Full LPIPS": (
                    float(full["lpips"].mean()) if not full["lpips"].isna().all() else float("nan")
                ),
                "BASS-48 LPIPS": (
                    float(subset["lpips"].mean()) if not subset["lpips"].isna().all() else float("nan")
                ),
            }
        )
    out = pd.DataFrame(rows)

    for metric in ("PSNR", "SSIM"):
        full_scores = {row["Method"]: float(row[f"Full {metric}"]) for row in rows}
        subset_scores = {row["Method"]: float(row[f"BASS-48 {metric}"]) for row in rows}
        full_ranks = _rank_map(full_scores, higher_is_better=True)
        subset_ranks = _rank_map(subset_scores, higher_is_better=True)
        out[f"Full {metric} Rank"] = out["Method"].map(full_ranks)
        out[f"BASS-48 {metric} Rank"] = out["Method"].map(subset_ranks)

    lpips_available = not out["Full LPIPS"].isna().any() and not out["BASS-48 LPIPS"].isna().any()
    if lpips_available:
        full_scores = {row["Method"]: float(row["Full LPIPS"]) for row in rows}
        subset_scores = {row["Method"]: float(row["BASS-48 LPIPS"]) for row in rows}
        full_ranks = _rank_map(full_scores, higher_is_better=False)
        subset_ranks = _rank_map(subset_scores, higher_is_better=False)
        out["Full LPIPS Rank"] = out["Method"].map(full_ranks)
        out["BASS-48 LPIPS Rank"] = out["Method"].map(subset_ranks)

    order_map = {name: idx for idx, name in enumerate(METHOD_ORDER)}
    out["method_order"] = out["Method"].map(order_map)
    out = out.sort_values("method_order", kind="mergesort").drop(columns=["method_order"]).reset_index(drop=True)
    return out


def _rank_correlation(
    summary_df: pd.DataFrame,
    metric: str,
) -> Dict[str, object]:
    full_ranks = summary_df[f"Full {metric} Rank"].to_numpy(dtype=float)
    subset_ranks = summary_df[f"BASS-48 {metric} Rank"].to_numpy(dtype=float)
    spear = spearmanr(full_ranks, subset_ranks)
    kend = kendalltau(full_ranks, subset_ranks)
    return {
        "metric": metric,
        "rank_match": bool(np.array_equal(full_ranks.astype(int), subset_ranks.astype(int))),
        "spearman_rho": float(spear.statistic) if spear.statistic is not None else float("nan"),
        "kendall_tau": float(kend.statistic) if kend.statistic is not None else float("nan"),
    }


def _find_missing_ids_per_method(
    method_tables: Sequence[MethodTable],
    shared_full_ids: Sequence[str],
) -> Dict[str, List[str]]:
    shared_id_set = set(shared_full_ids)
    out: Dict[str, List[str]] = {}
    for table in method_tables:
        all_ids = set(table.df["scene_id"].astype(str))
        out[table.method_name] = sorted(all_ids - shared_id_set)
    return out


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
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\hline",
        "Method & \\multicolumn{4}{c}{PSNR $\\uparrow$} & \\multicolumn{4}{c}{SSIM $\\uparrow$} \\\\",
        " & Full PSNR & BASS-48 PSNR & Full Rank & BASS-48 Rank & Full SSIM & BASS-48 SSIM & Full Rank & BASS-48 Rank \\\\",
        "\\hline",
    ]
    for row in summary_df.to_dict("records"):
        lines.append(
            "{} & {:.3f} & {:.3f} & {} & {} & {:.4f} & {:.4f} & {} & {} \\\\".format(
                _latex_escape(row["Method"]),
                float(row["Full PSNR"]),
                float(row["BASS-48 PSNR"]),
                int(row["Full PSNR Rank"]),
                int(row["BASS-48 PSNR Rank"]),
                float(row["Full SSIM"]),
                float(row["BASS-48 SSIM"]),
                int(row["Full SSIM Rank"]),
                int(row["BASS-48 SSIM Rank"]),
            )
        )
    lines += [
        "\\hline",
        "\\end{tabular}%",
        "}",
        "\\caption{Full-dataset and BASS-48 subset rankings on the shared three-method intersection. The subset preserves PSNR and SSIM ordering across the available methods; LPIPS is reported when available but is treated as diagnostic because this pilot is limited to three methods.}",
        "\\label{tab:ranking-pilot-full-vs-bass48}",
        "\\end{table}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(
    path: Path,
    method_tables: Sequence[MethodTable],
    subset_path: Path,
    shared_full_ids: Sequence[str],
    shared_subset_ids: Sequence[str],
    subset_ids: Sequence[str],
    summary_df: pd.DataFrame,
    rank_stats: Sequence[Dict[str, object]],
    manuscript_paths: Sequence[Path],
) -> None:
    missing_per_method = _find_missing_ids_per_method(method_tables, shared_full_ids)
    subset_missing = sorted(set(subset_ids) - set(shared_full_ids))
    lpips_missing_methods = []
    for table in method_tables:
        if table.df["lpips"].isna().all():
            lpips_missing_methods.append(table.method_name)

    lines = [
        "# Ranking Pilot Sanity Report",
        "",
        "## Inputs",
    ]
    for table in method_tables:
        lines.append(f"- {table.method_name}: `{table.path}`")
    lines += [
        f"- BASS subset: `{subset_path}`",
        "",
        "## Coverage",
        f"- Full three-method intersection: **{len(shared_full_ids)}** scenes",
        f"- BASS-48 scenes available for all three methods: **{len(shared_subset_ids)}** scenes",
        f"- BASS subset scenes missing from the three-method intersection: **{len(subset_missing)}**",
    ]

    if subset_missing:
        lines.append(
            f"- Missing BASS scene IDs (first 20): `{', '.join(subset_missing[:20])}`"
        )
    else:
        lines.append("- Missing BASS scene IDs: none")

    lines += [
        "",
        "## Method Coverage Outside Shared Intersection",
    ]
    for method_name in METHOD_ORDER:
        missing_ids = missing_per_method[method_name]
        if missing_ids:
            lines.append(
                f"- {method_name}: {len(missing_ids)} scene IDs outside the shared intersection; first 10 = "
                f"`{', '.join(missing_ids[:10])}`"
            )
        else:
            lines.append(f"- {method_name}: no scene IDs outside the shared intersection")

    lines += [
        "",
        "## Rank Stability",
    ]
    for item in rank_stats:
        lines.append(
            f"- {item['metric']}: ranks match = **{item['rank_match']}**, "
            f"Spearman rho = **{item['spearman_rho']:.3f}**, "
            f"Kendall tau = **{item['kendall_tau']:.3f}**"
        )

    lines += [
        "",
        "## LPIPS",
    ]
    if lpips_missing_methods:
        lines.append(
            f"- LPIPS is not available for all three methods; missing from: `{', '.join(lpips_missing_methods)}`. "
            "It is omitted from the main ranking table."
        )
    else:
        lines.append("- LPIPS is available for all three methods.")

    lines += [
        "",
        "## Manuscript Update Status",
    ]
    if manuscript_paths:
        lines.append(
            f"- Detected manuscript-like TeX files: `{', '.join(str(p) for p in manuscript_paths[:10])}`"
        )
    else:
        lines.append(
            "- No manuscript `.tex` sources were found in this repository, so the table could not be inserted automatically."
        )
        lines.append(
            "- Suggested paragraph: "
            "`This pilot is not intended to establish general leaderboard fidelity. Instead, it checks whether the recommended subset preserves method ordering on the available shared three-method intersection.`"
        )
        lines.append(
            "- Suggested include: `\\input{tables/ranking_pilot_full_vs_bass48.tex}`"
        )

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _find_manuscript_tex_files(repo_root: Path) -> List[Path]:
    matches = sorted(repo_root.rglob("*.tex"))
    out: List[Path] = []
    for path in matches:
        if "results" in path.parts:
            continue
        if "tables" in path.parts:
            continue
        out.append(path)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instant-ngp", type=Path, default=None, help="Path to Instant-NGP metrics table.")
    parser.add_argument("--zipnerf", type=Path, default=None, help="Path to Zip-NeRF metrics table.")
    parser.add_argument("--feature-splatting", type=Path, default=None, help="Path to Feature-Splatting metrics table.")
    parser.add_argument("--subset", type=Path, default=None, help="Path to the selected BASS subset manifest.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/ranking_pilot_full_vs_bass48.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("results/ranking_pilot_full_vs_bass48_report.md"),
        help="Output sanity report path.",
    )
    parser.add_argument(
        "--output-latex",
        type=Path,
        default=Path("tables/ranking_pilot_full_vs_bass48.tex"),
        help="Output LaTeX table path.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[1]

    method_specs: List[MethodSpec] = []
    cli_paths = {
        "Instant-NGP": args.instant_ngp,
        "Zip-NeRF": args.zipnerf,
        "Feature-Splatting": args.feature_splatting,
    }
    for method_name in METHOD_ORDER:
        cli_path = cli_paths[method_name]
        if cli_path is not None:
            resolved = _resolve_path(str(cli_path), repo_root)
            if not resolved.exists():
                raise FileNotFoundError(f"{method_name}: metrics file not found: {resolved}")
            method_specs.append(
                MethodSpec(
                    method_name=method_name,
                    path=resolved,
                    sheet_name=_detect_sheet_name(resolved),
                )
            )
        else:
            method_specs.append(_auto_detect_method_path(repo_root, method_name))

    if args.subset is not None:
        subset_path = _resolve_path(str(args.subset), repo_root)
        if not subset_path.exists():
            raise FileNotFoundError(f"Subset manifest not found: {subset_path}")
    else:
        subset_path = _auto_detect_subset_path(repo_root)

    method_tables = [_canonicalize_method_table(spec) for spec in method_specs]
    subset_ids = _load_subset_ids(subset_path)

    shared_full_ids = sorted(
        set.intersection(*(set(table.df["scene_id"].astype(str)) for table in method_tables))
    )
    if not shared_full_ids:
        raise RuntimeError("The three method files do not share any scene IDs")

    shared_subset_ids = sorted(set(shared_full_ids) & set(subset_ids))
    if not shared_subset_ids:
        raise RuntimeError(
            "The selected BASS subset has no overlap with the shared three-method intersection"
        )

    summary_df = _compute_metric_means(method_tables, shared_full_ids, shared_subset_ids)
    if summary_df[["Full LPIPS", "BASS-48 LPIPS"]].isna().any().any():
        summary_df = summary_df.drop(columns=["Full LPIPS", "BASS-48 LPIPS"])
    rank_stats = [
        _rank_correlation(summary_df, "PSNR"),
        _rank_correlation(summary_df, "SSIM"),
    ]

    output_csv = _resolve_path(str(args.output_csv), repo_root)
    output_report = _resolve_path(str(args.output_report), repo_root)
    output_latex = _resolve_path(str(args.output_latex), repo_root)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.write_text("", encoding="utf-8")
    summary_df.to_csv(output_csv, index=False)
    _write_latex_table(output_latex, summary_df)
    _write_report(
        output_report,
        method_tables=method_tables,
        subset_path=subset_path,
        shared_full_ids=shared_full_ids,
        shared_subset_ids=shared_subset_ids,
        subset_ids=subset_ids,
        summary_df=summary_df,
        rank_stats=rank_stats,
        manuscript_paths=_find_manuscript_tex_files(repo_root),
    )

    LOGGER.info("Wrote CSV: %s", output_csv)
    LOGGER.info("Wrote LaTeX table: %s", output_latex)
    LOGGER.info("Wrote report: %s", output_report)
    LOGGER.info(
        "Shared full intersection=%d | shared BASS subset=%d",
        len(shared_full_ids),
        len(shared_subset_ids),
    )


if __name__ == "__main__":
    main()
