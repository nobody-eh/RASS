#!/usr/bin/env python3
"""Task P17: DL3DV-140 instant-ngp logs on the local GPU (camera-ready).

Runs locally because instant-ngp needs tinycudann, which is installed here
but fails to build on the GPU cluster (its login node has no libcuda). The
cluster's pure-PyTorch fallback cost 106 min/scene, 5x splatfacto, so those
runs were quarantined as non-comparable rather than kept.

Scenes are streamed: download -> train -> eval -> delete, so peak disk stays
small. Downloads use direct resolve URLs and read the frame list from each
scene's transforms.json, so no Hub API calls are made (snapshot_download
enumerates the whole 2.1 TB repo tree and exhausts the request quota).

Usage: rebuttal_e17_local_instant_ngp.py <worker_id> <num_workers>
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRATCH = pathlib.Path(os.environ.get("RASS_SCRATCH", "/tmp/rass_scratch"))
WORK = SCRATCH / "dl3dv_ingp_work"
LOGS = SCRATCH / "dl3dv_ingp_logs"
META = REPO / "benchmark-meta.csv"
BASE = "https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark/resolve/main"
COLMAP = ["cameras.bin", "images.bin", "points3D.bin"]
METHOD = "instant-ngp"

# ~/.cache points at a full disk on this machine, so every cache torch or
# nerfstudio might touch is redirected here explicitly rather than being
# inherited from the launching shell.
for var, sub in (("TORCH_EXTENSIONS_DIR", "torch_ext"), ("TORCH_HOME", "torch"),
                 ("HF_HOME", "hf"), ("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "xdg")):
    os.environ.setdefault(var, str(SCRATCH / sub))
    (SCRATCH / sub).mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"


def fetch(rel: str, dst: pathlib.Path, tries: int = 5) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(tries):
        try:
            r = session.get(f"{BASE}/{rel}", stream=True, timeout=180)
            r.raise_for_status()
            tmp = dst.with_suffix(dst.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.rename(dst)
            return
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(20 * (attempt + 1))


def stage(h: str) -> pathlib.Path:
    ns = WORK / h / "nerfstudio"
    if (ns / ".complete").exists():
        return ns
    fetch(f"{h}/nerfstudio/transforms.json", ns / "transforms.json")
    meta = json.load(open(ns / "transforms.json"))
    names = [f["file_path"].split("/")[-1] for f in meta["frames"]]
    jobs = [(f"{h}/nerfstudio/images_4/{n}", ns / "images_4" / n) for n in names]
    jobs += [(f"{h}/nerfstudio/colmap/sparse/0/{c}",
              ns / "colmap/sparse/0" / c) for c in COLMAP]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda a: fetch(*a), jobs))
    (ns / ".complete").write_text(f"{len(names)} images\n")
    return ns


def main() -> None:
    global WORK
    worker = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nworkers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    WORK = WORK / f"w{worker}"          # per-worker: no cross-worker deletes
    (LOGS / "json").mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    hashes = pd.read_csv(META)["hash"].astype(str).tolist()
    env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")

    todo = [h for h in hashes if not (LOGS / "json" / f"{h}.json").exists()]
    todo = [h for i, h in enumerate(todo) if i % nworkers == worker]
    print(f"[w{worker}/{nworkers}] {len(todo)} scenes", flush=True)

    t0, ndone = time.time(), 0
    for i, h in enumerate(todo):
        try:
            ns_dir = stage(h)
            outdir = WORK / "train"
            shutil.rmtree(outdir / h, ignore_errors=True)
            r = subprocess.run(
                ["ns-train", METHOD, "--data", str(ns_dir),
                 "--output-dir", str(outdir), "--experiment-name", h,
                 "--vis", "tensorboard", "--machine.num-devices", "1",
                 "--machine.seed", "0",
                 "--viewer.quit-on-train-completion", "True",
                 "--steps-per-save", "30000",
                 "nerfstudio-data", "--downscale-factor", "4"],
                env=env, capture_output=True, text=True, timeout=14400,
                input="y\n",
            )
            ckpts = list((outdir / h).glob(
                f"{METHOD}/*/nerfstudio_models/step-000029999.ckpt"))
            if r.returncode != 0 or not ckpts:
                print(f"TRAIN FAIL {h[:12]}: {r.stderr[-300:]}", flush=True)
                continue
            out = LOGS / "json" / f"{h}.json"
            r = subprocess.run(
                ["ns-eval", "--load-config", str(ckpts[0].parent.parent / "config.yml"),
                 "--output-path", str(out)],
                env=env, capture_output=True, text=True, timeout=3600,
            )
            if r.returncode != 0 or not out.exists():
                print(f"EVAL FAIL {h[:12]}: {r.stderr[-300:]}", flush=True)
                continue
            d = json.load(open(out))
            d["method_name"] = METHOD
            d["experiment_name"] = h
            if "checkpoint" in d:
                d["checkpoint"] = pathlib.Path(d["checkpoint"]).name
            d["provenance"] = ("DL3DV-Benchmark scene, trained+evaluated locally on "
                               "RTX 3090 with tinycudann, ns-train instant-ngp "
                               "defaults, seed 0, downscale 4 (960P), 2026-07; "
                               "camera-ready material (P17)")
            json.dump(d, open(out, "w"), indent=2)
            ndone += 1
            total = len(list((LOGS / "json").glob("*.json")))
            rate = (time.time() - t0) / ndone / 60
            print(f"DONE {h[:12]} ({total}/140 total, {rate:.1f} min/scene, "
                  f"~{rate*(len(todo)-i-1)/60:.1f} h left)", flush=True)
        except Exception as exc:
            print(f"ERROR {h[:12]}: {str(exc)[:200]}", flush=True)
        finally:
            shutil.rmtree(WORK / h, ignore_errors=True)
            shutil.rmtree(WORK / "train" / h, ignore_errors=True)
    print(f"worker {worker} finished: {ndone} scenes, "
          f"{(time.time()-t0)/3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
