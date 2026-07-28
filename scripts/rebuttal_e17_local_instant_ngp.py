#!/usr/bin/env python3
"""Task P17: DL3DV-140 instant-ngp logs on the local GPU (camera-ready).

Runs locally because the GPU cluster is being retired and our queued job sits
behind other users. instant-ngp is the only method here that needs nerfacc's
JIT-compiled CUDA extension, which made it far more fragile than nerfacto,
splatfacto or tensorf (all 140/140 first time). Getting it running required:
a stale JIT lock cleared, CUDA_HOME corrected (the env pointed at a
non-existent 12.8 toolkit; the real nvcc is /usr/bin), MAX_JOBS=1 to survive
a memory-constrained machine, and LD_PRELOAD of the SYSTEM libstdc++ because
the extension is built with system g++ while conda ships an older
libstdc++ lacking GLIBCXX_3.4.32.

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
# DELIBERATE DEVIATION, must be disclosed wherever these logs are used:
# instant-ngp is trained to 16,000 steps while nerfacto/splatfacto/tensorf use
# the nerfstudio default of 30,000 (NVlabs' own run.py falls back to 35,000).
# Chosen because measured marginal train-PSNR gain per 2k steps drops below the
# DL3DV audit tolerance (0.2014 dB) at ~14-16k. The E17 scaling audit stays
# valid because its constraints are within-method, but no fair cross-method
# quality comparison can be drawn against the other three methods.
STEPS = 16000
CKPT = f"step-{STEPS - 1:09d}.ckpt"

# ~/.cache points at a full disk on this machine, so every cache torch or
# nerfstudio might touch is redirected here explicitly rather than being
# inherited from the launching shell.
for var, sub in (("TORCH_EXTENSIONS_DIR", "torch_ext"), ("TORCH_HOME", "torch"),
                 ("HF_HOME", "hf"), ("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "xdg")):
    os.environ.setdefault(var, str(SCRATCH / sub))
    (SCRATCH / sub).mkdir(parents=True, exist_ok=True)
# CUDA_HOME must be FORCED, not setdefault: the shell here exports a stale
# /usr/local/cuda-12.8 that does not exist, so setdefault silently keeps it
# and every nerfacc build dies with "nvcc: not found".
_cuda = next((d for d in ("/usr/local/cuda", "/usr")
              if os.path.exists(os.path.join(d, "bin", "nvcc"))), None)
if _cuda:
    os.environ["CUDA_HOME"] = _cuda
os.environ.setdefault("MAX_JOBS", "1")              # memory-constrained host
# match the arch of the verified prebuilt extension so torch reuses it
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
# reduce allocator fragmentation when two workers share one GPU
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_SYS_STDCXX = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if os.path.exists(_SYS_STDCXX):                     # conda libstdc++ is too old
    os.environ.setdefault("LD_PRELOAD", _SYS_STDCXX)

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
                 "--max-num-iterations", str(STEPS),
                 "--steps-per-save", str(STEPS),
                 # two workers share a 24 GB card; ns-eval renders full 960P
                 # images and OOMs against the other worker's training unless
                 # rendering is tiled. Chunking changes tiling only, not output.
                 "--pipeline.model.eval-num-rays-per-chunk", "2048",
                 "nerfstudio-data", "--downscale-factor", "4"],
                env=env, capture_output=True, text=True, timeout=14400,
                input="y\n",
            )
            ckpts = list((outdir / h).glob(
                f"{METHOD}/*/nerfstudio_models/{CKPT}"))
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
            d["provenance"] = (f"DL3DV-Benchmark scene, trained+evaluated locally on "
                               f"RTX 3090 with tinycudann, ns-train instant-ngp, "
                               f"seed 0, downscale 4 (960P), "
                               f"max_num_iterations={STEPS} (NOT the nerfstudio "
                               f"default of 30000 used by the other three methods; "
                               f"see E17 disclosure), 2026-07; camera-ready (P17)")
            d["train_steps"] = STEPS
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
