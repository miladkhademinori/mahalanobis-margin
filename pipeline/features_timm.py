"""pipeline.features_timm -- the recovered extraction protocol (timm, pooled pre-logits, timm's own eval transform).

Two transform conventions are implemented and the choice is recorded in every cache's meta:

  "timm"        (default) timm's own evaluation transform for the model, from
                timm.data.resolve_data_config(model.pretrained_cfg) + create_transform(is_training=False):
                bicubic resize to input_size / crop_pct, centre crop to input_size, the model's own mean and
                std (for vit_base_patch16_224.augreg_in21k: resize 248, crop 224, mean = std = 0.5).  The fleet's
                protocol matrix (letter of 2026-10-23, round 2) recovers the printed ViT gamma values under this
                transform (CIFAR-100 8.7 against 8.3, ImageNet-R 5.0 against 5.2) and not under the HF processor.
  "archaeology" the per-dataset transforms the owner's archaeology named: CIFAR Resize(224) + CIFAR statistics;
                CUB Resize(256) + CenterCrop(224) + ImageNet statistics; the other datasets as CUB.

The readout is model(x) with num_classes=0 (pooled pre-logits); for ViT the CLS token and the pooled readout
differ by under one percent under either transform (round 2), so the transform, not the readout, is the knob.
Caches are written with the same layout as `pipeline.features` into a SEPARATE root (default `features_timm/`),
so every downstream command runs on them with `--features features_timm` and the existing caches stay untouched.

  python replicate.py features-timm --data-root /content/data --out-root /content/drive/.../features_timm [--transform timm|archaeology]

timm model names (TIMM_IDS below) are the best-known equivalents of the paper's backbones; the owner's
script may name different tags (DINOv2 with registers, CLIP LAION-2B weights are the round-3 arms), and the
notebook's configuration cell is where they are replaced.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from pipeline import registry as R
from pipeline.features import build_dataset, cache_path, write_cache

TIMM_IDS: Dict[str, dict] = {
    "vit_in21k": dict(name="vit_base_patch16_224.augreg_in21k", pretrained=True),
    "dinov2_b14": dict(name="vit_base_patch14_dinov2.lvd142m", pretrained=True),
    "clip_b16": dict(name="vit_base_patch16_clip_224.openai", pretrained=True),
    "random_vit": dict(name="vit_base_patch16_224", pretrained=False, seed=0),
    "mae_b16": dict(name="vit_base_patch16_224.mae", pretrained=True),
    "resnet50": dict(name="resnet50.tv_in1k", pretrained=True),
}

CIFAR_MEAN, CIFAR_STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

TRANSFORMS: Dict[str, dict] = {
    "cifar100": dict(resize=224, crop=None, mean=CIFAR_MEAN, std=CIFAR_STD, note="Resize(224) + CIFAR-100 stats (archaeology)"),
    "cub200": dict(resize=256, crop=224, mean=IMAGENET_MEAN, std=IMAGENET_STD, note="Resize(256) + CenterCrop(224) + ImageNet stats (archaeology)"),
    "imagenet_r": dict(resize=256, crop=224, mean=IMAGENET_MEAN, std=IMAGENET_STD, note="assumed as CUB (not named by the archaeology)"),
    "omnibenchmark": dict(resize=256, crop=224, mean=IMAGENET_MEAN, std=IMAGENET_STD, note="assumed as CUB (not named by the archaeology)"),
    "imagenet100": dict(resize=256, crop=224, mean=IMAGENET_MEAN, std=IMAGENET_STD, note="ImageNet-style"),
    "domainnet": dict(resize=256, crop=224, mean=IMAGENET_MEAN, std=IMAGENET_STD, note="ImageNet-style"),
}


def build_transform(dataset: str):
    """The archaeology's per-dataset transform (convention "archaeology")."""
    from torchvision import transforms as T
    t = TRANSFORMS[dataset]; ops = []
    if t["crop"]:
        ops += [T.Resize(t["resize"]), T.CenterCrop(t["crop"])]
    else:
        ops += [T.Resize((t["resize"], t["resize"]))]
    ops += [T.ToTensor(), T.Normalize(t["mean"], t["std"])]
    return T.Compose(ops)


def build_timm_transform(model, img_size: int = 224):
    """timm's own evaluation transform for ``model`` (convention "timm"), and a dict describing it for the meta."""
    import timm.data
    cfg = timm.data.resolve_data_config({}, model=model)
    cfg = dict(cfg); cfg["input_size"] = (3, img_size, img_size)
    tf = timm.data.create_transform(**cfg, is_training=False)
    desc = dict(input_size=list(cfg["input_size"]), interpolation=cfg.get("interpolation"), crop_pct=cfg.get("crop_pct"),
                crop_mode=cfg.get("crop_mode"), mean=[float(v) for v in cfg.get("mean", ())], std=[float(v) for v in cfg.get("std", ())],
                resize_to=int(np.floor(img_size / float(cfg.get("crop_pct") or 1.0))), note="timm resolve_data_config + create_transform(is_training=False)",
                repr=repr(tf))
    return tf, desc


def load_timm_backbone(backbone: str, device: str = "cuda", ids: Optional[Dict[str, dict]] = None, img_size: int = 224):
    """timm model with num_classes=0 (pooled pre-logits) at ``img_size``; returns (model, info)."""
    import timm
    import torch
    spec = (ids or TIMM_IDS)[backbone]
    if not spec.get("pretrained", True):
        torch.manual_seed(int(spec.get("seed", 0)))
    kwargs = dict(pretrained=spec.get("pretrained", True), num_classes=0)
    try:
        model = timm.create_model(spec["name"], img_size=img_size, **kwargs)
    except TypeError:                     # models without an img_size argument (ResNet)
        model = timm.create_model(spec["name"], **kwargs)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    info = dict(timm=timm.__version__, name=spec["name"], pretrained=bool(spec.get("pretrained", True)),
                global_pool=str(getattr(model, "global_pool", "")), num_features=int(getattr(model, "num_features", 0)),
                pretrained_cfg=dict(input_size=cfg.get("input_size"), mean=cfg.get("mean"), std=cfg.get("std"), crop_pct=cfg.get("crop_pct"),
                                    interpolation=cfg.get("interpolation"), hf_hub_id=cfg.get("hf_hub_id")), img_size=int(img_size),
                forward="model(x) with num_classes=0: forward_head(forward_features(x), pre_logits=True)")
    return model, info


def determinism_check_timm(backbone: str, device: str = "cuda", ids=None) -> dict:
    import torch
    model, info = load_timm_backbone(backbone, device, ids)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(8, 3, 224, 224, generator=g).to(device)
    with torch.no_grad():
        a = model(x).float().cpu().numpy(); b = model(x).float().cpu().numpy()
    return dict(backbone=backbone, name=info["name"], bit_identical=bool(np.array_equal(a, b)), max_abs_diff=float(np.abs(a - b).max()),
                sha256=hashlib.sha256(a.tobytes()).hexdigest()[:16])


def extract_timm(backbone: str, dataset: str, data_root: Path, out_root: Path, *, batch_size: int = 128, device: str = "cuda",
                 num_workers: int = 4, imagenet100_root: Optional[Path] = None, fp16: bool = False, ids=None,
                 transform: str = "timm", img_size: int = 224, log=print) -> Dict[str, Path]:
    """Extract and cache train and test features under the recovered protocol. Idempotent per split.
    ``transform`` is "timm" (the model's own eval transform; default) or "archaeology" (per-dataset, see TRANSFORMS).
    fp16 is off by default: the fingerprint comparison is sensitive at the percent level."""
    import torch
    out_root = Path(out_root)
    done = {s: cache_path(backbone, dataset, s, out_root) for s in ("train", "test")}
    if all(p.exists() for p in done.values()):
        log(f"[features-timm] {backbone}/{dataset}: cache present, skipping")
        return done
    splits = build_dataset(dataset, data_root, imagenet100_root)
    model, info = load_timm_backbone(backbone, device, ids, img_size=img_size)
    if transform == "timm":
        tf, tf_desc = build_timm_transform(model, img_size)
    elif transform == "archaeology":
        tf, tf_desc = build_transform(dataset), dict(TRANSFORMS[dataset])
    else:
        raise ValueError(f"transform must be 'timm' or 'archaeology', got {transform!r}")
    meta_base = dict(backbone=backbone, dataset=dataset, protocol=f"timm pooled pre-logits, transform={transform}", timm_model=info,
                     transform_convention=transform, transform=tf_desc, torch=torch.__version__, fp16=bool(fp16), provenance="real",
                     device=str(torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else device))

    class _DS(torch.utils.data.Dataset):
        def __init__(self, il): self.il = il
        def __len__(self): return len(self.il)
        def __getitem__(self, i): return tf(self.il.image(i)), i

    for split, il in splits.items():
        if done[split].exists():
            continue
        loader = torch.utils.data.DataLoader(_DS(il), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))
        feats = None; t0 = time.time()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(fp16 and device == "cuda")):
            for pv, idx in loader:
                f = model(pv.to(device, non_blocking=True)).float().cpu().numpy()
                if feats is None:
                    feats = np.zeros((len(il), f.shape[1]), np.float32)
                feats[idx.numpy()] = f
        meta = meta_base | dict(split=split, classes=il.classes, seconds=round(time.time() - t0, 1))
        write_cache(done[split], feats, il.labels, meta, il.domains)
        log(f"[features-timm] {backbone}/{dataset}/{split}: {len(il)} images, d={feats.shape[1]}, {time.time()-t0:.0f}s -> {done[split]}")
    return done


def manifest(out_root: Path) -> Path:
    """sha256 of every cache file under out_root, for the readback record."""
    out_root = Path(out_root); rows = []
    for p in sorted(out_root.rglob("*.npz")):
        rows.append(dict(path=str(p.relative_to(out_root)), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size))
    mp = out_root / "MANIFEST.json"; mp.write_text(json.dumps(dict(root=str(out_root), files=rows, written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=1))
    return mp


def paper_a_configs():
    return [(b, d) for b in list(R.PAPER_A_BACKBONES) + ["random_vit"] for d in R.PAPER_A_DATASETS]
