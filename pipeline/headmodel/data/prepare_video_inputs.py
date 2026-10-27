#!/usr/bin/env python3
"""Prepare registered frame metadata, SAM masks, and the A-stage overlay sheet."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from render_overlay import load_registered_frame, render_overlay_sheet


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from segment_references import FACE_ROIS, JAWS, SPECS  # noqa: E402
from stage1_sam3_segment import call_sam3_on_crop  # noqa: E402


DATA_DIR = Path(__file__).resolve().parent
REGISTRATION_PATH = REPO_ROOT / "source/viewer/ponytail-seedance-registration.json"
FRAME_DIR = WORKSPACE_ROOT / ".local/artifacts/20261025-ponytail-seedance-reg/frames"
WIDTH = 720
HEIGHT = 720
CELL = 360
SAMPLE_STEP = 3
HEAD_PROMPTS = {"head": "head", "hair_bun": "hair bun", "ponytail": "ponytail"}
EAR_PROMPT = "human ear"
FACE_PROMPT = "face skin of anime girl excluding ears neck and hair"
FRONT_FACE_SPEC = "left-face"
EAR_LABELS = (("ear-left", 0), ("ear-right", 1))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_registration() -> dict[str, object]:
    registration = json.loads(REGISTRATION_PATH.read_text(encoding="utf-8"))
    video = registration["video"]
    if len(video) != 361:
        raise ValueError(f"Expected 361 registered frames, got {len(video)}")
    return registration


def build_frame_index(registration: dict[str, object]) -> list[dict[str, object]]:
    key_rows = {int(row["sourceFrame"]): row for row in registration["keys"]}
    selected = set(range(1, 362, SAMPLE_STEP)) | set(key_rows)
    rows = []
    for frame_number in sorted(selected):
        video_row = registration["video"][frame_number - 1]
        pose = [float(value) for value in video_row["pose"]]
        if len(pose) != 6:
            raise ValueError(f"Frame {frame_number} does not have a six-value pose")
        key_row = key_rows.get(frame_number)
        row: dict[str, object] = {
            "frame": frame_number,
            "role": "key" if key_row else "sample",
            "pose": pose,
            "has_normalization": bool(key_row and key_row.get("normalization")),
        }
        if key_row:
            row["key_name"] = key_row["name"]
        rows.append(row)
    return rows


def binary_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def fill_holes_and_keep_largest(mask: np.ndarray) -> np.ndarray:
    flood = mask.copy()
    flood_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    filled = cv2.bitwise_or(mask, cv2.bitwise_not(flood))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def clip_to_jaw(mask: np.ndarray, jaw_points: list[list[int]]) -> np.ndarray:
    clip = np.zeros_like(mask)
    polygon = np.asarray(
        [(0, 0), (0, jaw_points[0][1]), *jaw_points, (WIDTH - 1, jaw_points[-1][1]), (WIDTH - 1, 0)],
        dtype=np.int32,
    )
    cv2.fillPoly(clip, [polygon], 255)
    return cv2.bitwise_and(mask, clip)


def bbox_from_contour(contour: list[list[float]]) -> list[int] | None:
    if not contour:
        return None
    points = np.asarray(contour, dtype=np.float32)
    x0 = max(0, int(np.floor(points[:, 0].min())) - 10)
    y0 = max(0, int(np.floor(points[:, 1].min())) - 10)
    x1 = min(WIDTH, int(np.ceil(points[:, 0].max())) + 11)
    y1 = min(HEIGHT, int(np.ceil(points[:, 1].max())) + 11)
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask.shape != (HEIGHT, WIDTH) or mask.dtype != np.uint8:
        raise ValueError(f"Mask must be uint8 {WIDTH}x{HEIGHT}: {path}")
    if not cv2.imwrite(str(path), mask):
        raise RuntimeError(f"Could not write mask: {path}")


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def skin_spec(direction: str) -> tuple[tuple[int, int, int, int], str, list[tuple[int, int, int]] | None, list[tuple[int, int, int, int]] | None]:
    if direction in FACE_ROIS:
        crop, points = FACE_ROIS[direction]
        return crop, FACE_PROMPT, points, None
    spec_key = "right-face" if direction == "right" else "left-face"
    if direction == "front":
        spec_key = FRONT_FACE_SPEC
    _, crop, prompt, points, boxes = SPECS[spec_key]
    return crop, prompt, points, boxes


def main() -> None:
    registration = read_registration()
    frame_rows = build_frame_index(registration)
    save_json(
        DATA_DIR / "frames.json",
        {
            "source": "source/viewer/ponytail-seedance-registration.json",
            "frame_directory": str(FRAME_DIR),
            "source_fps": 24,
            "source_size": [640, 640],
            "normalized_size": [WIDTH, HEIGHT],
            "sample_step": SAMPLE_STEP,
            "frames": frame_rows,
        },
    )

    key_rows = {int(row["sourceFrame"]): row for row in registration["keys"]}
    selected_keys = [row for row in frame_rows if row["role"] == "key"]
    retry_order = (42, 240, 206, 270, 192, 219, 303, 264, 3)
    processing_keys = sorted(
        selected_keys,
        key=lambda row: retry_order.index(int(row["frame"])) if int(row["frame"]) in retry_order else len(retry_order),
    )
    missing = [int(row["frame"]) for row in selected_keys if not (FRAME_DIR / f"frame_{int(row['frame']):04d}.png").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing selected key frames: {missing}")

    sys.path.insert(0, str(Path(os.environ["AGENT_COMMON_FILES_DIR"]) / "lib"))
    from secret_loader import get_secret

    fal_key = get_secret("shared/fal", "api_key")
    sam_module = sys.modules[call_sam3_on_crop.__module__]
    original_post = sam_module.requests.post
    post_attempts = 0

    def counted_post(*args: object, **kwargs: object):
        nonlocal post_attempts
        post_attempts += 1
        return original_post(*args, **kwargs)

    sam_module.requests.post = counted_post
    total_logical_calls = 0
    failure_streak = 0
    width_results: dict[int, int] = {}
    key_lookup = {int(row["sourceFrame"]): row for row in registration["keys"]}

    try:
        for frame_row in processing_keys:
            frame_number = int(frame_row["frame"])
            key_row = key_lookup[frame_number]
            rgb, normalized = load_registered_frame(frame_number, key_row)
            source_path = FRAME_DIR / f"frame_{frame_number:04d}.png"
            image = Image.fromarray(rgb).convert("RGBA")
            frame_dir = DATA_DIR / f"frame_{frame_number:04d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            source_sha256 = sha256_file(source_path)
            manifest_path = frame_dir / "manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("source_image_sha256") != source_sha256:
                    raise ValueError(f"Cached manifest source frame changed: {frame_number}")
            else:
                manifest = {
                    "frame": frame_number,
                    "key_name": key_row["name"],
                    "role": "key",
                    "pose": frame_row["pose"],
                    "source_image_sha256": source_sha256,
                    "normalization_applied": normalized,
                    "masks": {},
                    "calls": [],
                    "endpoint": "https://fal.run/fal-ai/sam-3/image",
                }
            masks_manifest = manifest["masks"]

            def request_mask(
                mask_name: str,
                prompt: str,
                crop: tuple[int, int, int, int],
                points: list[tuple[int, int, int]] | None = None,
                boxes: list[tuple[int, int, int, int]] | None = None,
                upscale: int = 1,
                empty_is_valid: bool = False,
            ) -> tuple[np.ndarray | None, dict[str, object]]:
                nonlocal total_logical_calls, failure_streak
                total_logical_calls += 1
                start_attempts = post_attempts
                called_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                started = time.perf_counter()
                try:
                    result_mask, metadata = call_sam3_on_crop(
                        fal_key,
                        image,
                        crop,
                        prompt,
                        points=points,
                        boxes=boxes,
                        upscale=upscale,
                    )
                    failure_streak = 0
                    record = {
                        "mask": mask_name,
                        "prompt": prompt,
                        "called_at_utc": called_at,
                        "duration_seconds": round(time.perf_counter() - started, 3),
                        "http_post_attempts": post_attempts - start_attempts,
                        **metadata,
                        "success": True,
                    }
                    manifest["calls"].append(record)
                    save_json(manifest_path, manifest)
                    return binary_mask(result_mask), record
                except ValueError as error:
                    if empty_is_valid and str(error) == "SAM 3 returned an empty mask list":
                        failure_streak = 0
                        record = {
                            "mask": mask_name,
                            "prompt": prompt,
                            "called_at_utc": called_at,
                            "duration_seconds": round(time.perf_counter() - started, 3),
                            "http_post_attempts": post_attempts - start_attempts,
                            "success": True,
                            "visible": False,
                            "empty_result": True,
                            "score": None,
                            "area_px": 0,
                        }
                        manifest["calls"].append(record)
                        save_json(manifest_path, manifest)
                        return np.zeros((HEIGHT, WIDTH), dtype=np.uint8), record
                    failure_streak += 1
                    record = {
                        "mask": mask_name,
                        "prompt": prompt,
                        "called_at_utc": called_at,
                        "duration_seconds": round(time.perf_counter() - started, 3),
                        "http_post_attempts": post_attempts - start_attempts,
                        "success": False,
                        "error_type": type(error).__name__,
                    }
                    manifest["calls"].append(record)
                    save_json(manifest_path, manifest)
                    if frame_number == 42:
                        raise RuntimeError("SAM 3 request failed on probe frame 42") from None
                    if failure_streak >= 3:
                        raise RuntimeError("Three consecutive SAM 3 calls failed") from None
                    return None, record
                except Exception as error:
                    failure_streak += 1
                    record = {
                        "mask": mask_name,
                        "prompt": prompt,
                        "called_at_utc": called_at,
                        "duration_seconds": round(time.perf_counter() - started, 3),
                        "http_post_attempts": post_attempts - start_attempts,
                        "success": False,
                        "error_type": type(error).__name__,
                    }
                    manifest["calls"].append(record)
                    save_json(manifest_path, manifest)
                    if frame_number == 42:
                        raise RuntimeError("SAM 3 request failed on probe frame 42") from None
                    if failure_streak >= 3:
                        raise RuntimeError("Three consecutive SAM 3 calls failed") from None
                    return None, record

            full_crop = (0, 0, WIDTH, HEIGHT)
            intermediates: dict[str, np.ndarray | None] = {}
            existing_head = masks_manifest.get("head", {})
            head_file = frame_dir / str(existing_head.get("file", "head.png"))
            if existing_head.get("success") and head_file.is_file():
                head = cv2.imread(str(head_file), cv2.IMREAD_GRAYSCALE)
                head_success = head is not None and bool(cv2.countNonZero(head))
                if head is None:
                    head = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            else:
                for mask_name, prompt in HEAD_PROMPTS.items():
                    mask, _ = request_mask(
                        mask_name,
                        prompt,
                        full_crop,
                        empty_is_valid=mask_name in ("hair_bun", "ponytail"),
                    )
                    intermediates[mask_name] = mask
                if all(intermediates[name] is not None for name in HEAD_PROMPTS):
                    head = cv2.bitwise_and(intermediates["head"], cv2.bitwise_not(intermediates["hair_bun"]))
                    head = cv2.bitwise_and(head, cv2.bitwise_not(intermediates["ponytail"]))
                    head = fill_holes_and_keep_largest(head)
                    head_success = bool(cv2.countNonZero(head))
                else:
                    head = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                    head_success = False
                write_mask(frame_dir / "head.png", head)
                masks_manifest["head"] = {
                    "file": "head.png",
                    "success": head_success,
                    "area_px": int(cv2.countNonZero(head)),
                    "source_prompts": HEAD_PROMPTS,
                    "postprocess": ["subtract hair bun", "subtract ponytail", "fill holes", "keep largest component"],
                }
                if frame_number == 42 and not head_success:
                    manifest["head_bbox_width_px"] = 0
                    save_json(manifest_path, manifest)
                    raise RuntimeError("Head mask remained empty on probe frame 42") from None

            existing_skin = masks_manifest.get("skin", {})
            skin_path = frame_dir / str(existing_skin.get("file", "skin.png"))
            if existing_skin.get("success") and skin_path.is_file():
                skin = cv2.imread(str(skin_path), cv2.IMREAD_GRAYSCALE)
                if skin is None:
                    skin = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            else:
                crop, skin_prompt, skin_points, skin_boxes = skin_spec(str(key_row["name"]))
                skin_raw, skin_call = request_mask("skin", skin_prompt, crop, skin_points, skin_boxes)
                if skin_raw is None:
                    skin = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                    skin_success = False
                else:
                    skin = clip_to_jaw(skin_raw, JAWS[str(key_row["name"])]) if str(key_row["name"]) in JAWS else skin_raw
                    skin_success = bool(cv2.countNonZero(skin))
                write_mask(frame_dir / "skin.png", skin)
                masks_manifest["skin"] = {
                    "file": "skin.png",
                    "success": skin_success,
                    "score": skin_call.get("score"),
                    "area_px": int(cv2.countNonZero(skin)),
                    "prompt": skin_prompt,
                    "jaw_clip_applied": str(key_row["name"]) in JAWS,
                }

            for mask_name, contour_index in EAR_LABELS:
                existing_ear = masks_manifest.get(mask_name, {})
                existing_ear_path = frame_dir / str(existing_ear.get("file", f"{mask_name}.png"))
                if existing_ear.get("success") and existing_ear_path.is_file():
                    continue
                contour = key_row["earContours"][contour_index]
                box = bbox_from_contour(contour)
                if box is None:
                    ear_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                    ear_record = {
                        "file": f"{mask_name}.png",
                        "visible": False,
                        "success": True,
                        "score": None,
                        "area_px": 0,
                        "bbox": None,
                        "prompt": EAR_PROMPT,
                    }
                else:
                    ear_mask, ear_call = request_mask(
                        mask_name,
                        EAR_PROMPT,
                        tuple(box),
                        boxes=[tuple(box)],
                        upscale=2,
                        empty_is_valid=True,
                    )
                    if ear_mask is None:
                        ear_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                        ear_success = False
                    else:
                        ear_success = True
                    ear_visible = None if not ear_success else bool(cv2.countNonZero(ear_mask))
                    ear_record = {
                        "file": f"{mask_name}.png",
                        "visible": ear_visible,
                        "success": ear_success,
                        "score": ear_call.get("score"),
                        "area_px": int(cv2.countNonZero(ear_mask)),
                        "bbox": box,
                        "prompt": EAR_PROMPT,
                    }
                write_mask(frame_dir / f"{mask_name}.png", ear_mask)
                masks_manifest[mask_name] = ear_record

            nonzero = cv2.findNonZero(head)
            points = None if nonzero is None else nonzero.reshape(-1, 2)
            width = 0 if points is None else int(points[:, 0].max() - points[:, 0].min() + 1)
            manifest["head_bbox_width_px"] = width
            width_results[frame_number] = width
            save_json(manifest_path, manifest)
            print(f"frame {frame_number}: {key_row['name']} complete; width={width}px", flush=True)
    finally:
        sam_module.requests.post = original_post

    front_width = width_results.get(3, 0)
    wider_frames = [
        {"frame": frame, "width_px": width, "ratio_to_front_3": round(width / front_width, 3)}
        for frame, width in sorted(width_results.items())
        if front_width and width > front_width * 1.3
    ]
    sheet = render_overlay_sheet(registration, selected_keys)
    mask_successes = {name: 0 for name in ("head", "skin", "ear-left", "ear-right")}
    mask_total = {name: 0 for name in mask_successes}
    for frame_row in selected_keys:
        frame_dir = DATA_DIR / f"frame_{int(frame_row['frame']):04d}"
        manifest = json.loads((frame_dir / "manifest.json").read_text(encoding="utf-8"))
        for name in mask_successes:
            mask_total[name] += 1
            if manifest["masks"][name]["success"]:
                mask_successes[name] += 1
    print(f"logical SAM prompts: {total_logical_calls}; HTTP POST attempts: {post_attempts}")
    print(f"mask success: {mask_successes} / {mask_total}")
    print(f"head width > 1.3x front: {[row['frame'] for row in wider_frames]}")
    print(f"overlay sheet: {sheet}")


if __name__ == "__main__":
    main()
