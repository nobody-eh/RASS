#!/usr/bin/env python3
"""Task P6 part b / key E6.b: DINOv2 descriptor variant.

Per scene: sample 8 frames uniformly (deterministic linspace indices),
DINOv2 ViT-S/14 CLS embeddings, mean-pool, L2-normalize -> one 384-D vector
per scene. KMeans k=6 seed 0 directly on the L2-normalized vectors
(declared choice; embeddings are already on a comparable scale).
Balanced Zip-NeRF audit over the standard budget grid, M=400, seed 0.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault(
    "TORCH_HOME",
    os.path.join(os.environ.get("RASS_SCRATCH", "/tmp/rass_scratch"), "torch"),
)

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_strong_accept_results import (  # noqa: E402
    _load_zipnerf_metrics,
    _merge_mapping_and_metrics,
)
from rebuttal_audit_tasks import merge_results_json  # noqa: E402
from rebuttal_e3_sensitivity import FEATS, ZIPNERF_LOG, Sweeper  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "rebuttal"
SCENE_ROOTS = [
    REPO / "n5k360p/n5k360l/360_4",
    REPO / "n5k360p/n5k360l/360_3",
    REPO / "n5k360p/n5k360l/360_2",
]
EMB_CACHE = OUT_DIR / "e6b_dinov2_embeddings.npz"
N_FRAMES = 8
BUDGETS = [4, 6, 8, 10, 12, 14, 16, 20]
P_MIN = 0.08

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def frame_paths(scene_dir: Path) -> list[Path]:
    img_dir = scene_dir / "images"
    if not img_dir.is_dir():
        return []
    frames = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    if not frames:
        return []
    idx = np.linspace(0, len(frames) - 1, N_FRAMES).astype(int)
    return [frames[i] for i in idx]


def load_batch(paths: list[Path]) -> torch.Tensor:
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)
        t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
        imgs.append((t - MEAN) / STD)
    return torch.stack(imgs)


def main() -> None:
    feats_ids = pd.read_csv(FEATS, usecols=["dish_id"])["dish_id"].astype(str).tolist()
    scene_dirs = {}
    for root in SCENE_ROOTS:
        if not root.is_dir():
            continue
        for d in root.iterdir():
            scene_dirs.setdefault(d.name, d)
    targets = [(i, scene_dirs[i]) for i in feats_ids if i in scene_dirs]
    missing = [i for i in feats_ids if i not in scene_dirs]
    print(f"scenes with frames: {len(targets)} / {len(feats_ids)} (missing {len(missing)})")

    done: dict[str, np.ndarray] = {}
    if EMB_CACHE.exists():
        data = np.load(EMB_CACHE, allow_pickle=True)
        done = dict(zip(data["ids"].tolist(), data["embs"]))
        print(f"resuming: {len(done)} scenes already embedded", flush=True)
    todo = [(i, d) for i, d in targets if i not in done]
    skipped = []
    valid = []
    for dish_id, sdir in todo:
        paths = frame_paths(sdir)
        if len(paths) < N_FRAMES:
            skipped.append(dish_id)
        else:
            valid.append((dish_id, paths))
    if valid:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        model.eval().to(device)

        class SceneDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(valid)

            def __getitem__(self, i):
                dish_id, paths = valid[i]
                return dish_id, load_batch(paths)  # [8, 3, 224, 224]

        loader = torch.utils.data.DataLoader(
            SceneDataset(), batch_size=8, num_workers=8,
            pin_memory=(device == "cuda"),
        )

        def save():
            k = list(done.keys())
            np.savez_compressed(EMB_CACHE, ids=np.array(k), embs=np.stack([done[i] for i in k]))

        n_done = 0
        with torch.no_grad():
            for dish_ids, batches in loader:  # batches: [B, 8, 3, 224, 224]
                b = batches.shape[0]
                flat = batches.view(b * N_FRAMES, *batches.shape[2:]).to(device, non_blocking=True)
                cls = model(flat).view(b, N_FRAMES, -1)  # [B, 8, 384]
                v = cls.mean(dim=1)
                v = v / v.norm(dim=1, keepdim=True)
                v = v.cpu().numpy()
                for j, dish_id in enumerate(dish_ids):
                    done[str(dish_id)] = v[j]
                n_done += b
                if n_done % 400 < 8:
                    save()
                    print(f"  {n_done}/{len(valid)} scenes embedded (checkpointed)", flush=True)
        save()
        print(f"embedded total {len(done)}; skipped {len(skipped)} with <8 frames", flush=True)
    ids = list(done.keys())
    embs = np.stack([done[i] for i in ids])

    labels = KMeans(n_clusters=6, random_state=0, n_init="auto").fit_predict(embs)
    mapping = pd.DataFrame({"dish_id": ids, "cluster": labels})
    merged = _merge_mapping_and_metrics(mapping, _load_zipnerf_metrics(ZIPNERF_LOG))
    sw = Sweeper(merged)
    sizes = {c: len(g) for c, g in sw.groups.items()}
    print(f"audit population {len(merged)}; regime sizes {sizes}")
    bs = [b for b in BUDGETS if b <= min(sizes.values())]
    frontier, rec = sw.recommended_budget(bs, audit_seed=0)
    for r in frontier:
        print(f"  {r['budget_scenes']:3d} scenes: {r['n_pass']:3d}/400 "
              f"(LCB {r['wilson_lcb_95']:.4f})")
    print(f"recommended budget at 0.08: {rec}")

    e6b = {
        "status": "completed",
        "model": "dinov2_vits14 (torch.hub facebookresearch/dinov2)",
        "protocol": (
            "8 uniformly-sampled frames per scene (deterministic linspace), "
            "224x224 bilinear, ImageNet normalization, CLS embeddings "
            "mean-pooled then L2-normalized; KMeans k=6 seed 0 (n_init="
            "'auto') directly on the L2-normalized 384-D vectors; balanced "
            "Zip-NeRF audit, paper tolerances, M=400, audit seed 0, "
            "rng=default_rng(seed+b)."
        ),
        "n_scenes_embedded": int(len(ids)),
        "n_scenes_missing_frames": int(len(feats_ids) - len(ids)),
        "regime_sizes": {str(k): v for k, v in sizes.items()},
        "audit_population": int(len(merged)),
        "frontier": frontier,
        "budget_p008": rec if rec is not None else "not reached up to 120",
        "embeddings_cache": str(EMB_CACHE.relative_to(REPO)),
    }
    merged_json = OUT_DIR / "rebuttal_results.json"
    import json
    existing = json.loads(merged_json.read_text())
    e6 = existing.get("E6", {})
    e6["b"] = e6b
    merge_results_json(merged_json, {"E6": e6})
    print("Wrote E6.b")


if __name__ == "__main__":
    main()
