#!/usr/bin/env python3
"""Task P17 / key E17: streamed per-scene DL3DV-140 splatfacto (3DGS) logs.

Camera-ready material: second DL3DV method committed in the rebuttal.
Same streamed design as the E13 nerfacto pipeline (download -> train ->
eval -> delete; resumable, scene done == JSON exists). Descriptors already
exist from E13, so no descriptor stage. splatfacto defaults, seed 0,
downscale 4 (960P). Runs at low priority relative to interactive work.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRATCH = pathlib.Path(os.environ.get("RASS_SCRATCH", "/tmp/rass_scratch"))
WORK = SCRATCH / "dl3dv_splat_work"
LOGS = SCRATCH / "dl3dv_splatfacto_logs"
META = REPO / "benchmark-meta.csv"
REPO_ID = "DL3DV/DL3DV-Benchmark"

os.environ.setdefault("HF_HOME", str(SCRATCH / "hf"))


def download_scene(h: str) -> pathlib.Path:
    from huggingface_hub import snapshot_download
    dst = WORK / h
    snapshot_download(
        REPO_ID, repo_type="dataset", local_dir=str(dst),
        allow_patterns=[f"{h}/nerfstudio/transforms.json",
                        f"{h}/nerfstudio/images_4/*",
                        f"{h}/nerfstudio/colmap/sparse/0/cameras.bin",
                        f"{h}/nerfstudio/colmap/sparse/0/images.bin",
                        f"{h}/nerfstudio/colmap/sparse/0/points3D.bin"],
        token=os.environ.get("HF_TOKEN"),
    )
    return dst / h / "nerfstudio"


def main() -> None:
    worker = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nworkers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    WORK.mkdir(parents=True, exist_ok=True)
    (LOGS / "json").mkdir(parents=True, exist_ok=True)
    hashes = pd.read_csv(META)["hash"].tolist()
    env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    todo = [h for h in hashes if not (LOGS / "json" / f"{h}.json").exists()]
    todo = [h for i, h in enumerate(todo) if i % nworkers == worker]
    print(f"[w{worker}/{nworkers}] {len(todo)} scenes to process", flush=True)

    prefetched: dict[str, pathlib.Path] = {}

    def prefetch(h):
        try:
            prefetched[h] = download_scene(h)
        except Exception as exc:
            print(f"PREFETCH FAIL {h[:12]}: {exc}", flush=True)

    t0 = time.time()
    ndone = 0
    for i, h in enumerate(todo):
        try:
            ns_dir = prefetched.pop(h, None) or download_scene(h)
            if i + 1 < len(todo):
                threading.Thread(target=prefetch, args=(todo[i + 1],), daemon=True).start()

            outdir = WORK / "train"
            shutil.rmtree(outdir / h, ignore_errors=True)
            r = subprocess.run(
                ["ns-train", "splatfacto", "--data", str(ns_dir),
                 "--output-dir", str(outdir), "--experiment-name", h,
                 "--vis", "tensorboard", "--machine.num-devices", "1",
                 "--machine.seed", "0",
                 "--viewer.quit-on-train-completion", "True",
                 "nerfstudio-data", "--downscale-factor", "4"],
                env=env, capture_output=True, text=True, timeout=10800,
            )
            ckpts = list((outdir / h).glob("splatfacto/*/nerfstudio_models/step-000029999.ckpt"))
            if r.returncode != 0 or not ckpts:
                print(f"TRAIN FAIL {h[:12]}: {r.stderr[-300:]}", flush=True)
                continue
            run_dir = ckpts[0].parent.parent
            out = LOGS / "json" / f"{h}.json"
            r = subprocess.run(
                ["ns-eval", "--load-config", str(run_dir / "config.yml"),
                 "--output-path", str(out)],
                env=env, capture_output=True, text=True, timeout=3600,
            )
            if r.returncode != 0 or not out.exists():
                print(f"EVAL FAIL {h[:12]}: {r.stderr[-300:]}", flush=True)
                continue
            d = json.load(open(out))
            d["method_name"] = "splatfacto"
            d["experiment_name"] = h
            d["provenance"] = ("DL3DV-Benchmark scene, trained+evaluated locally on RTX 3090, "
                               "ns-train splatfacto defaults, seed 0, downscale 4 (960P), "
                               "2026-07; camera-ready material (P17)")
            json.dump(d, open(out, "w"), indent=2)
            ndone += 1
            total_json = len(list((LOGS / "json").glob("*.json")))
            rate = (time.time() - t0) / ndone / 60
            print(f"DONE {h[:12]} ({total_json}/140 total, {rate:.1f} min/scene, "
                  f"~{rate*(len(todo)-i-1)/60:.1f} h left)", flush=True)
        except Exception as exc:
            print(f"ERROR {h[:12]}: {exc}", flush=True)
        finally:
            shutil.rmtree(WORK / h, ignore_errors=True)
            shutil.rmtree(WORK / "train" / h, ignore_errors=True)
    print(f"pipeline finished: {len(list((LOGS/'json').glob('*.json')))}/140 JSONs "
          f"({(time.time()-t0)/3600:.1f} h)", flush=True)


if __name__ == "__main__":
    main()
