"""SAM response retrieval and raw-mask storage for head-model inputs."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent.parent
DATA_DIR = Path(__file__).resolve().parent
FRAME_DIR = WORKSPACE_ROOT / ".local/artifacts/20261025-ponytail-seedance-reg/frames"
WIDTH = 720
HEIGHT = 720
MASK_REVISION = "a-stage-mask-revision-2"
PROMPTS = ("head", "hair bun", "ponytail", "face skin", "ear", "neck")
REUSABLE_OLD_PROMPTS = {"head", "hair bun", "ponytail"}

sys.path.insert(0, str(REPO_ROOT / "pipeline"))
from stage1_sam3_segment import SAM3_ENDPOINT, image_to_data_uri  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def union_masks(masks: list[np.ndarray]) -> np.ndarray:
    union = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for mask in masks:
        if mask.shape != (HEIGHT, WIDTH):
            raise ValueError(f"SAM mask has unexpected shape: {mask.shape}")
        union = cv2.bitwise_or(union, binary_mask(mask))
    return union


def fill_holes_and_keep_largest(mask: np.ndarray) -> np.ndarray:
    flood = mask.copy()
    flood_mask = np.zeros((HEIGHT + 2, WIDTH + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    filled = cv2.bitwise_or(mask, cv2.bitwise_not(flood))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask.shape != (HEIGHT, WIDTH) or mask.dtype != np.uint8:
        raise ValueError(f"Mask must be uint8 {WIDTH}x{HEIGHT}: {path}")
    if not cv2.imwrite(str(path), mask):
        raise RuntimeError(f"Could not write mask: {path}")


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_prompt_cache(
    prompt: str,
    frame_dir: Path,
    raw_manifest: dict[str, dict[str, object]],
    source_sha256: str,
) -> list[np.ndarray] | None:
    info = raw_manifest.get(prompt)
    raw_path = frame_dir / "raw" / f"{prompt}.png"
    if not raw_path.is_file():
        return None
    if info and (
        info.get("prompt") != prompt
        or info.get("source_image_sha256") not in (None, source_sha256)
        or info.get("success") is False
    ):
        return None
    if prompt not in REUSABLE_OLD_PROMPTS and (
        not info
        or info.get("mask_revision") != MASK_REVISION
        or info.get("source_image_sha256") != source_sha256
    ):
        return None
    raw_union = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
    if raw_union is None or raw_union.shape != (HEIGHT, WIDTH):
        return None
    if info and int(info.get("num_instances", 0)) == 0:
        return []
    if prompt == "ear" and info:
        instances = []
        for relative_path in info.get("instance_files", []):
            instance_path = frame_dir / str(relative_path)
            instance = cv2.imread(str(instance_path), cv2.IMREAD_GRAYSCALE)
            if instance is None or instance.shape != (HEIGHT, WIDTH):
                return None
            instances.append(binary_mask(instance))
        if instances:
            return instances
    return [binary_mask(raw_union)] if cv2.countNonZero(raw_union) else []


def load_sam_instances(
    fal_key: str,
    image: Image.Image,
    prompt: str,
) -> tuple[list[np.ndarray], dict[str, object], int]:
    background = Image.new("RGBA", image.size, (238, 240, 245, 255))
    background.alpha_composite(image.convert("RGBA"))
    payload = {
        "image_url": image_to_data_uri(background.convert("RGB")),
        "prompt": prompt,
        "apply_mask": False,
        "return_multiple_masks": True,
        "max_masks": 4,
        "include_scores": True,
    }
    post_attempts = 0
    for attempt in range(3):
        post_attempts += 1
        response = requests.post(
            SAM3_ENDPOINT,
            headers={
                "Authorization": f"Key {fal_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            break
        if attempt < 2:
            time.sleep(1.5 * (attempt + 1))
    else:
        raise RuntimeError(f"SAM 3 failed with HTTP {response.status_code}")

    data = response.json()
    returned = data.get("masks") or []
    scores = data.get("scores") or []
    instances = []
    for entry in returned:
        download = requests.get(entry["url"], timeout=60)
        download.raise_for_status()
        mask_image = Image.open(io.BytesIO(download.content)).convert("L")
        if mask_image.size != (WIDTH, HEIGHT):
            mask_image = mask_image.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)
        instances.append(binary_mask(np.asarray(mask_image)))
    score_list = [
        float(scores[index]) if index < len(scores) and scores[index] is not None else None
        for index in range(len(returned))
    ]
    metadata = {
        "prompt": prompt,
        "num_masks_returned": len(returned),
        "scores": score_list,
        "instance_area_px": [int(cv2.countNonZero(mask)) for mask in instances],
    }
    return instances, metadata, post_attempts


def persist_raw_prompt(
    prompt: str,
    instances: list[np.ndarray],
    frame_dir: Path,
    raw_manifest: dict[str, dict[str, object]],
    source_sha256: str,
    metadata: dict[str, object],
    called_at: str,
    duration: float,
) -> None:
    raw_dir = frame_dir / "raw"
    raw_path = raw_dir / f"{prompt}.png"
    write_mask(raw_path, union_masks(instances))
    instance_files: list[str] = []
    if prompt == "ear":
        instance_dir = raw_dir / "ear"
        if instance_dir.exists():
            shutil.rmtree(instance_dir)
        for index, mask in enumerate(instances):
            relative_path = f"raw/ear/instance_{index:02d}.png"
            write_mask(frame_dir / relative_path, mask)
            instance_files.append(relative_path)
    raw_manifest[prompt] = {
        "file": f"raw/{prompt}.png",
        "prompt": prompt,
        "success": True,
        "mask_revision": MASK_REVISION,
        "source_image_sha256": source_sha256,
        "called_at_utc": called_at,
        "duration_seconds": round(duration, 3),
        "num_instances": len(instances),
        "scores": metadata["scores"],
        "instance_area_px": metadata["instance_area_px"],
        "instance_files": instance_files,
    }


def prompt_center_x(mask: np.ndarray) -> float | None:
    _, xs = np.nonzero(mask)
    return float(xs.mean()) if len(xs) else None
