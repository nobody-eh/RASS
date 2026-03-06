import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, rel_path: str):
    module_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYSIS = _load_module("run_feature_analysis_pipeline", "scripts/run_feature_analysis_pipeline.py")
EXTRACT = _load_module("build_feature_report_pipeline", "scripts/build_feature_report_pipeline.py")


class TestFeaturePipeline(unittest.TestCase):
    def test_canonicalize_metric_columns_fi_schema(self):
        fi_df = pd.DataFrame(
            {
                "dish_id": ["10", "11"],
                "PNSR": [25.0, 26.5],
                "MIN": [20.0, 21.0],
                "MAX": [30.0, 31.0],
                "SSIM": [0.80, 0.85],
                "MIN.1": [0.70, 0.75],
                "MAX.1": [0.90, 0.95],
                "psnr avgmse": [24.0, 25.2],
            }
        )

        out = ANALYSIS._canonicalize_metric_columns(fi_df, "fi")

        self.assertListEqual(list(out.columns), ["dish_id"] + ANALYSIS.METRIC_COLUMNS)
        self.assertAlmostEqual(float(out.loc[0, "PSNR"]), 25.0)
        self.assertAlmostEqual(float(out.loc[0, "PSNR_MIN"]), 20.0)
        self.assertAlmostEqual(float(out.loc[0, "PSNR_MAX"]), 30.0)
        self.assertAlmostEqual(float(out.loc[0, "SSIM"]), 0.80)
        self.assertAlmostEqual(float(out.loc[0, "SSIM_MIN"]), 0.70)
        self.assertAlmostEqual(float(out.loc[0, "SSIM_MAX"]), 0.90)
        self.assertAlmostEqual(float(out.loc[0, "psnr_avgmse"]), 24.0)

    def test_build_feature_table_deterministic_schema_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scene_b = root / "scene_b"
            scene_a = root / "scene_a"
            for scene in (scene_b, scene_a):
                (scene / "sparse" / "txt").mkdir(parents=True)
                (scene / "transforms_train.json").write_text("{}", encoding="utf-8")
                (scene / "sparse" / "txt" / "images.txt").write_text("#", encoding="utf-8")
                (scene / "sparse" / "txt" / "points3D.txt").write_text("#", encoding="utf-8")

            fake_records = [
                {
                    "dish_id": "scene_b",
                    "scene_path": str(scene_b),
                    "z_feat": "3.0",
                    "a_feat": "1.0",
                },
                {
                    "dish_id": "scene_a",
                    "scene_path": str(scene_a),
                    "z_feat": "2.0",
                    "a_feat": "0.0",
                },
            ]

            with patch.object(EXTRACT, "_run_parallel_extraction", return_value=(fake_records, [])):
                df, failures, _cache, _hits = EXTRACT.build_feature_table(
                    [scene_b / "transforms_train.json", scene_a / "transforms_train.json"],
                    num_workers=1,
                    show_progress=False,
                    max_frames_per_scene=None,
                    frame_stride=1,
                    resume=False,
                    cache_path=root / "cache.json",
                    mask_dirs=["masks_omvs", "masks", "rgba"],
                    require_mask=False,
                )

            self.assertEqual(len(failures), 0)
            self.assertListEqual(list(df.columns), ["dish_id", "scene_path", "a_feat", "z_feat"])
            self.assertListEqual(df["dish_id"].tolist(), ["scene_a", "scene_b"])
            self.assertTrue(np.issubdtype(df["a_feat"].dtype, np.number))
            self.assertTrue(np.issubdtype(df["z_feat"].dtype, np.number))

    def test_compute_drift_without_baseline_corr_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_summary_path = root / "baseline_summary.json"
            baseline_summary = {
                "per_feature_stats": {
                    "f1": {"mean": 10.0, "missing_percent": 1.0},
                }
            }
            baseline_summary_path.write_text(json.dumps(baseline_summary), encoding="utf-8")

            current_summary = {
                "per_feature_stats": {
                    "f1": {"mean": 11.0, "missing_percent": 2.0},
                }
            }
            current_corr = pd.DataFrame([[1.0]], index=["f1"], columns=["f1"])

            drift = EXTRACT._compute_drift(
                current_summary,
                current_corr,
                baseline_summary_path=baseline_summary_path,
                baseline_corr_path=None,
                mean_delta_threshold=0.1,
                missing_delta_threshold=0.1,
            )

            self.assertIsNone(drift["baseline_corr_csv"])
            self.assertIn("f1", drift["changed_features"])
            self.assertIsNone(drift["correlation_drift"])


if __name__ == "__main__":
    unittest.main()
