#!/usr/bin/env python3
"""Local (RTX 3090) training+evaluation of nerfacto scenes missing from the
log set. Runs as N independent workers (worker w handles indices i % N == w).

Per scene: shadow-prep (validated recipe) -> ns-train nerfacto
(--machine.num-devices 1, otherwise cluster flags) -> ns-eval -> JSON merged into
the harvested log dir (with local provenance note) -> training output and
shadow deleted to protect disk. Idempotent: scenes with an existing JSON are
skipped, so this composes with the queued cluster jobs (whoever finishes first
wins).

Priority order: regimes 0,1,2,4 first (the coverage-skewed ones), then 3,5.
Usage: rebuttal_local_nerfacto_train.py <worker_id> <num_workers>
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rebuttal_local_nerfacto_eval import OUT_JSON, SCRATCH, find_scene, prep_shadow  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
TRAIN_ROOT = SCRATCH / "local_train"
K6 = REPO / "sweep_cluster_k/k_6/clustered_scenes_k6_dish_cluster_mapping.csv"
PRIORITY = {0: 0, 1: 0, 2: 0, 4: 0, 3: 1, 5: 1}


def missing_scenes() -> list[str]:
    mapping = pd.read_csv(K6)
    have = {p.stem for p in OUT_JSON.glob("*.json")}
    rows = mapping[~mapping["dish_id"].isin(have)].copy()
    rows["prio"] = rows["cluster"].map(PRIORITY)
    rows = rows.sort_values(["prio", "dish_id"], kind="mergesort")
    return rows["dish_id"].tolist()


def main() -> None:
    worker, nworkers = int(sys.argv[1]), int(sys.argv[2])
    env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    done = failed = 0
    t0 = time.time()
    scenes = missing_scenes()
    print(f"[w{worker}] {len(scenes)} scenes missing; handling ~{len(scenes)//nworkers}", flush=True)
    for i, dish in enumerate(scenes):
        if i % nworkers != worker:
            continue
        out = OUT_JSON / f"{dish}.json"
        if out.exists():
            continue
        src = find_scene(dish)
        if src is None:
            print(f"[w{worker}] NO DATA {dish}", flush=True)
            continue
        try:
            sh = prep_shadow(dish, src)
            outdir = TRAIN_ROOT / f"w{worker}"
            if (outdir / dish).exists():
                shutil.rmtree(outdir / dish)
            r = subprocess.run(
                ["ns-train", "nerfacto", "--data", str(sh), "--output-dir", str(outdir),
                 "--vis", "tensorboard", "--machine.num-devices", "1",
                 "--viewer.quit-on-train-completion", "True"],
                env=env, capture_output=True, text=True, timeout=3600,
            )
            ckpts = list((outdir / dish).glob("nerfacto/*/nerfstudio_models/step-000029999.ckpt"))
            if r.returncode != 0 or not ckpts:
                print(f"[w{worker}] TRAIN FAIL {dish}: {r.stderr[-200:]}", flush=True)
                failed += 1
                continue
            run_dir = ckpts[0].parent.parent
            cfg = yaml.load(open(run_dir / "config.yml"), Loader=yaml.Loader)
            cfg.data = sh
            cfg.pipeline.datamanager.data = sh
            cfg.pipeline.datamanager.dataparser.data = sh
            cfg.load_dir = None
            yaml.dump(cfg, open(run_dir / "config_local.yml", "w"))
            r = subprocess.run(
                ["ns-eval", "--load-config", str(run_dir / "config_local.yml"),
                 "--output-path", str(out)],
                env=env, capture_output=True, text=True, timeout=1800,
            )
            if r.returncode != 0 or not out.exists():
                print(f"[w{worker}] EVAL FAIL {dish}: {r.stderr[-200:]}", flush=True)
                failed += 1
                continue
            d = json.load(open(out))
            d["provenance"] = ("trained+evaluated locally on RTX 3090, ns-train nerfacto "
                               "--machine.num-devices 1 (cluster protocol, single-GPU), 2026-07")
            json.dump(d, open(out, "w"), indent=2)
            done += 1
            rate = (time.time() - t0) / max(done, 1) / 60
            print(f"[w{worker}] DONE {dish} ({done} total, {rate:.1f} min/scene)", flush=True)
        except Exception as exc:
            print(f"[w{worker}] ERROR {dish}: {exc}", flush=True)
            failed += 1
        finally:
            shutil.rmtree(TRAIN_ROOT / f"w{worker}" / dish, ignore_errors=True)
            for extra in ("images_2", "images_4", "images_8"):
                shutil.rmtree(SCRATCH / "scene_shadow" / dish / extra, ignore_errors=True)
    print(f"[w{worker}] finished: {done} done, {failed} failed "
          f"({(time.time()-t0)/3600:.1f} h)", flush=True)


if __name__ == "__main__":
    main()
