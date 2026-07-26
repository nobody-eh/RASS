#!/usr/bin/env python3
"""Local (RTX 3090) evaluation of the 188 trained-but-unevaluated nerfacto
scenes, replacing the queued cluster job 43780697 if it stays pending.

Pipeline per scene (validated on dish_1550704750: local vs cluster metrics agree
to 3e-4 dB PSNR / 2e-5 SSIM / 2e-5 LPIPS):
1. shadow scene dir in scratchpad (images symlinked; scene data verified
   identical to cluster by checksum): pycolmap sparse/0 -> sparse/0_txt, then
   the cluster tooling archive's colmap2nerf.py --keep_colmap_coords -> transforms.json.
2. patch the run's config.yml paths (data -> shadow, output_dir -> local
   extraction root), keeping dataparser settings untouched.
3. ns-eval (TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 for torch 2.7) -> JSON in the
   same schema/location convention as cluster output_json_nerfacto/json.

Idempotent: skips scenes whose JSON already exists (locally produced or
arriving from the cluster job, whichever finishes first).
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRATCH = pathlib.Path(os.environ.get("RASS_SCRATCH", "/tmp/rass_scratch"))
ROOT = SCRATCH / "nerfacto_ckpts/root"
SHADOWS = SCRATCH / "scene_shadow"
OUT_JSON = SCRATCH / "cluster_logs/output_json_nerfacto/json"  # merge with harvested logs
COLMAP2NERF = SCRATCH / "cluster_code/src/colmap2nerf.py"
SCENE_ROOTS = [
    REPO / "n5k360p/n5k360l/360_4",
    REPO / "n5k360p/n5k360l/360_3",
    REPO / "n5k360p/n5k360l/360_2",
]


def find_scene(dish: str) -> pathlib.Path | None:
    for r in SCENE_ROOTS:
        p = r / dish
        if p.is_dir():
            return p
    return None


def prep_shadow(dish: str, src: pathlib.Path) -> pathlib.Path:
    sh = SHADOWS / dish
    (sh / "sparse").mkdir(parents=True, exist_ok=True)
    img_link = sh / "images"
    if not img_link.exists():
        img_link.symlink_to(src / "images")
    txt = sh / "sparse/0_txt"
    if not (txt / "images.txt").exists():
        txt.mkdir(parents=True, exist_ok=True)
        import pycolmap
        pycolmap.Reconstruction(str(src / "sparse/0")).write_text(str(txt))
    tj = sh / "transforms.json"
    if not tj.exists():
        subprocess.run(
            [sys.executable, str(COLMAP2NERF), "--colmap_camera_model", "SIMPLE_RADIAL",
             "--images", str(sh), "--text", str(txt), "--out", str(tj),
             "--keep_colmap_coords"],
            check=True, capture_output=True, text=True, cwd=str(sh),
        )
    return sh


def main() -> None:
    OUT_JSON.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(ROOT.glob("gpfs/**/nerfstudio_models/step-000029999.ckpt"))
    print(f"checkpoints available: {len(run_dirs)}", flush=True)
    done = skipped = failed = 0
    t0 = time.time()
    env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    for n, ckpt in enumerate(run_dirs):
        run_dir = ckpt.parent.parent
        dish = next(p for p in run_dir.parts if p.startswith("dish_"))
        out = OUT_JSON / f"{dish}.json"
        if out.exists():
            skipped += 1
            continue
        src = find_scene(dish)
        if src is None or not (run_dir / "config.yml").exists():
            print(f"  MISSING inputs for {dish}", flush=True)
            failed += 1
            continue
        try:
            sh = prep_shadow(dish, src)
            cfg = yaml.load(open(run_dir / "config.yml"), Loader=yaml.Loader)
            cfg.data = sh
            cfg.pipeline.datamanager.data = sh
            cfg.pipeline.datamanager.dataparser.data = sh
            # output_dir such that get_checkpoint_dir() == run_dir/nerfstudio_models
            cfg.output_dir = run_dir.parent.parent.parent
            cfg.load_dir = None
            with open(run_dir / "config_local.yml", "w") as f:
                yaml.dump(cfg, f)
            r = subprocess.run(
                ["ns-eval", "--load-config", str(run_dir / "config_local.yml"),
                 "--output-path", str(out)],
                env=env, capture_output=True, text=True, timeout=1800,
            )
            if r.returncode != 0 or not out.exists():
                print(f"  FAIL {dish}: {r.stderr[-300:]}", flush=True)
                failed += 1
                continue
            done += 1
            if done % 10 == 0:
                rate = (time.time() - t0) / max(done, 1)
                print(f"  {done} evaluated ({rate:.0f}s/scene, "
                      f"~{rate*(len(run_dirs)-n-1)/60:.0f} min left)", flush=True)
        except Exception as exc:
            print(f"  ERROR {dish}: {exc}", flush=True)
            failed += 1
    print(f"finished: {done} evaluated, {skipped} already had JSON, {failed} failed "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
