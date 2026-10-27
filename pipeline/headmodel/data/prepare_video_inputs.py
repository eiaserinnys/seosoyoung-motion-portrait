#!/usr/bin/env python3
"""Prepare registered frames and revisioned SAM masks for head-model alignment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from render_overlay import (  # noqa: E402
    load_registered_frame,
    render_overlay_sheet,
    render_raw_sheet,
)
from sam_mask_ops import (  # noqa: E402
    DATA_DIR,
    FRAME_DIR,
    HEIGHT,
    MASK_REVISION,
    PROMPTS,
    SAM3_ENDPOINT,
    WIDTH,
    binary_mask,
    fill_holes_and_keep_largest,
    load_sam_instances,
    persist_raw_prompt,
    prompt_center_x,
    read_prompt_cache,
    save_json,
    sha256_file,
    union_masks,
    write_mask,
)


REGISTRATION_PATH = REPO_ROOT / "source/viewer/ponytail-seedance-registration.json"
SAMPLE_STEP = 3



def read_registration() -> dict[str, object]:
    registration = json.loads(REGISTRATION_PATH.read_text(encoding="utf-8"))
    if len(registration["video"]) != 361:
        raise ValueError(f"Expected 361 registered frames, got {len(registration['video'])}")
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


def process_frame(
    frame_row: dict[str, object],
    key_row: dict[str, object],
    fal_key: str,
    failure_state: dict[str, int],
) -> dict[str, object]:
    frame_number = int(frame_row["frame"])
    source_path = FRAME_DIR / f"frame_{frame_number:04d}.png"
    rgb, normalized = load_registered_frame(frame_number, key_row)
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
            "endpoint": SAM3_ENDPOINT,
        }
    manifest["mask_definition_revision"] = MASK_REVISION
    manifest["normalization_applied"] = normalized
    raw_manifest = manifest.setdefault("raw_masks", {})
    masks_manifest = manifest.setdefault("masks", {})
    (frame_dir / "raw").mkdir(parents=True, exist_ok=True)
    instances_by_prompt: dict[str, list[np.ndarray]] = {}
    raw_success: dict[str, bool] = {}

    for prompt in PROMPTS:
        cached = read_prompt_cache(prompt, frame_dir, raw_manifest, source_sha256)
        if cached is not None:
            instances_by_prompt[prompt] = cached
            raw_success[prompt] = True
            if prompt in raw_manifest:
                raw_manifest[prompt]["success"] = True
            continue
        called_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        started = time.perf_counter()
        try:
            instances, metadata, post_attempts = load_sam_instances(fal_key, image, prompt)
            duration = time.perf_counter() - started
            persist_raw_prompt(
                prompt,
                instances,
                frame_dir,
                raw_manifest,
                source_sha256,
                metadata,
                called_at,
                duration,
            )
            record = {
                **metadata,
                "mask": prompt,
                "mask_revision": MASK_REVISION,
                "called_at_utc": called_at,
                "duration_seconds": round(duration, 3),
                "http_post_attempts": post_attempts,
                "success": True,
            }
            manifest.setdefault("calls", []).append(record)
            instances_by_prompt[prompt] = instances
            raw_success[prompt] = True
            failure_state["consecutive"] = 0
        except Exception as error:
            duration = time.perf_counter() - started
            record = {
                "mask": prompt,
                "prompt": prompt,
                "mask_revision": MASK_REVISION,
                "called_at_utc": called_at,
                "duration_seconds": round(duration, 3),
                "success": False,
                "error_type": type(error).__name__,
            }
            manifest.setdefault("calls", []).append(record)
            raw_manifest[prompt] = {
                "prompt": prompt,
                "mask_revision": MASK_REVISION,
                "source_image_sha256": source_sha256,
                "called_at_utc": called_at,
                "success": False,
                "error_type": type(error).__name__,
            }
            instances_by_prompt[prompt] = []
            raw_success[prompt] = False
            failure_state["consecutive"] += 1
            save_json(manifest_path, manifest)
            print(f"frame {frame_number}: prompt {prompt!r} failed", flush=True)
            if frame_number == 3 or failure_state["consecutive"] >= 3:
                raise RuntimeError(
                    f"SAM 3 failed on frame {frame_number}; stopping this run"
                ) from None
        save_json(manifest_path, manifest)

    prompt_unions = {prompt: union_masks(instances) for prompt, instances in instances_by_prompt.items()}
    head_raw = prompt_unions["head"]
    hair_bun_raw = prompt_unions["hair bun"]
    ponytail_raw = prompt_unions["ponytail"]
    neck_raw = prompt_unions["neck"]
    head = cv2.bitwise_and(head_raw, cv2.bitwise_not(hair_bun_raw))
    head = cv2.bitwise_and(head, cv2.bitwise_not(ponytail_raw))
    if len(instances_by_prompt["neck"]):
        head = cv2.bitwise_and(head, cv2.bitwise_not(neck_raw))
    head = fill_holes_and_keep_largest(head)
    skin_raw = prompt_unions["face skin"]
    skin = fill_holes_and_keep_largest(skin_raw)
    head_success = raw_success["head"] and bool(cv2.countNonZero(head))
    skin_success = raw_success["face skin"] and bool(cv2.countNonZero(skin))
    write_mask(frame_dir / "head.png", head)
    write_mask(frame_dir / "skin.png", skin)

    head_center_x = prompt_center_x(head)
    ear_instances = instances_by_prompt["ear"]
    ear_candidates: dict[str, list[dict[str, object]]] = {"ear-left": [], "ear-right": []}
    discarded_ears: list[dict[str, object]] = []
    ear_scores = raw_manifest.get("ear", {}).get("scores", [])
    for index, instance in enumerate(ear_instances):
        area = int(cv2.countNonZero(instance))
        center_x = prompt_center_x(instance)
        overlap = cv2.bitwise_and(instance, head)
        overlap_px = int(cv2.countNonZero(overlap))
        detail: dict[str, object] = {
            "instance_index": index,
            "score": ear_scores[index] if index < len(ear_scores) else None,
            "area_px": area,
            "center_x": center_x,
            "overlap_with_head_px": overlap_px,
        }
        if not area or center_x is None or overlap_px == 0 or head_center_x is None:
            detail["discard_reason"] = "no overlap with head" if overlap_px == 0 else "empty instance"
            discarded_ears.append(detail)
            continue
        side = "ear-left" if center_x < head_center_x else "ear-right"
        detail["side"] = side
        ear_candidates[side].append(detail)

    ear_results = {}
    for side, candidates in ear_candidates.items():
        selected_index = None
        if candidates:
            selected = max(candidates, key=lambda item: int(item["area_px"]))
            selected_index = int(selected["instance_index"])
        selected_mask = (
            binary_mask(ear_instances[selected_index])
            if selected_index is not None
            else np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        )
        write_mask(frame_dir / f"{side}.png", selected_mask)
        ear_results[side] = {
            "file": f"{side}.png",
            "success": raw_success["ear"],
            "visible": selected_index is not None,
            "area_px": int(cv2.countNonZero(selected_mask)),
            "selected_instance_index": selected_index,
            "candidates": candidates,
            "discarded_instances": discarded_ears,
            "head_center_x": head_center_x,
            "prompt": "ear",
        }
    masks_manifest["head"] = {
        "file": "head.png",
        "success": head_success,
        "area_px": int(cv2.countNonZero(head)),
        "source_prompts": {
            "head": "head",
            "hair_bun": "hair bun",
            "ponytail": "ponytail",
            "neck": "neck",
        },
        "neck_returned": bool(instances_by_prompt["neck"]),
        "postprocess": [
            "subtract hair bun",
            "subtract ponytail",
            "subtract neck when returned",
            "fill holes",
            "keep largest component",
        ],
    }
    masks_manifest["skin"] = {
        "file": "skin.png",
        "success": skin_success,
        "area_px": int(cv2.countNonZero(skin)),
        "prompt": "face skin",
        "postprocess": ["union all returned instances", "fill holes", "keep largest component"],
    }
    masks_manifest["ear-left"] = ear_results["ear-left"]
    masks_manifest["ear-right"] = ear_results["ear-right"]

    head_points = cv2.findNonZero(head)
    head_coordinates = None if head_points is None else head_points.reshape(-1, 2)
    head_width = 0 if head_coordinates is None else int(
        head_coordinates[:, 0].max() - head_coordinates[:, 0].min() + 1
    )
    manifest["head_bbox_width_px"] = head_width
    manifest["neck_returned"] = bool(instances_by_prompt["neck"])
    save_json(manifest_path, manifest)
    print(
        f"frame {frame_number}: head={head_success}, skin={skin_success}, "
        f"ear-visible=({ear_results['ear-left']['visible']},"
        f"{ear_results['ear-right']['visible']}), neck={bool(instances_by_prompt['neck'])}",
        flush=True,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames",
        help="Comma-separated key frame numbers to process, for example 3 or 42,192,206",
    )
    parser.add_argument("--final-sheets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registration = read_registration()
    frame_rows = build_frame_index(registration)
    key_rows = {int(row["sourceFrame"]): row for row in registration["keys"]}
    selected_keys = [row for row in frame_rows if row["role"] == "key"]
    if args.frames:
        frame_numbers = {int(value) for value in args.frames.split(",")}
        selected = [row for row in selected_keys if int(row["frame"]) in frame_numbers]
        if {int(row["frame"]) for row in selected} != frame_numbers:
            raise ValueError(f"Requested frames are not registered keys: {sorted(frame_numbers)}")
    else:
        selected = selected_keys

    missing = [
        int(row["frame"])
        for row in selected
        if not (FRAME_DIR / f"frame_{int(row['frame']):04d}.png").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing selected key frames: {missing}")
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

    sys.path.insert(0, str(Path(os.environ["AGENT_COMMON_FILES_DIR"]) / "lib"))
    from secret_loader import get_secret

    fal_key = get_secret("shared/fal", "api_key")
    failure_state = {"consecutive": 0}
    try:
        for frame_row in selected:
            manifest = process_frame(
                frame_row,
                key_rows[int(frame_row["frame"])],
                fal_key,
                failure_state,
            )
            if int(frame_row["frame"]) == 3 and (
                not manifest["masks"]["head"]["success"]
                or not manifest["masks"]["skin"]["success"]
            ):
                raise RuntimeError("Frame 3 head or skin mask is empty; stopping after probe")
    except Exception:
        render_raw_sheet(registration, selected)
        raise

    if args.final_sheets:
        expected = {int(row["frame"]) for row in selected_keys}
        available = {
            int(row["frame"])
            for row in selected_keys
            if (DATA_DIR / f"frame_{int(row['frame']):04d}" / "manifest.json").is_file()
        }
        if expected - available:
            raise RuntimeError(f"Cannot render final sheets; missing key frames: {sorted(expected - available)}")
        for row in selected_keys:
            manifest_path = DATA_DIR / f"frame_{int(row['frame']):04d}" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            head = cv2.imread(str(manifest_path.parent / "head.png"), cv2.IMREAD_GRAYSCALE)
            points = cv2.findNonZero(head) if head is not None else None
            coordinates = None if points is None else points.reshape(-1, 2)
            width = 0 if coordinates is None else int(
                coordinates[:, 0].max() - coordinates[:, 0].min() + 1
            )
            manifest["head_bbox_width_px"] = width
            save_json(manifest_path, manifest)
        overlay = render_overlay_sheet(registration, selected_keys)
        raw_sheet = render_raw_sheet(registration, selected_keys)

        manifests = [
            json.loads(
                (DATA_DIR / f"frame_{int(row['frame']):04d}" / "manifest.json").read_text(encoding="utf-8")
            )
            for row in selected_keys
        ]
        front_width = next(
            (int(item.get("head_bbox_width_px", 0)) for item in manifests if int(item["frame"]) == 3),
            0,
        )
        wider = [
            {
                "frame": int(item["frame"]),
                "width_px": int(item.get("head_bbox_width_px", 0)),
                "ratio_to_front_3": round(int(item.get("head_bbox_width_px", 0)) / front_width, 3),
            }
            for item in manifests
            if front_width and int(item.get("head_bbox_width_px", 0)) > front_width * 1.3
        ]
        revision_calls = [
            call
            for item in manifests
            for call in item.get("calls", [])
            if call.get("mask_revision") == MASK_REVISION
        ]
        mask_successes = {
            name: sum(bool(item["masks"][name]["success"]) for item in manifests)
            for name in ("head", "skin", "ear-left", "ear-right")
        }
        ear_visibility = {
            side: sum(bool(item["masks"][side]["visible"]) for item in manifests)
            for side in ("ear-left", "ear-right")
        }
        ear_not_visible = {side: len(manifests) - count for side, count in ear_visibility.items()}
        neck_frames = sum(bool(item.get("neck_returned")) for item in manifests)
        neck_instances = sum(
            int(item.get("raw_masks", {}).get("neck", {}).get("num_instances", 0))
            for item in manifests
        )
        print(f"mask success: {mask_successes} / {len(manifests)} frames")
        print(f"ear visible true: {ear_visibility}; false: {ear_not_visible}")
        print(f"neck returned: {neck_frames}/{len(manifests)} frames; {neck_instances} instances")
        print(
            f"SAM revision calls: {len(revision_calls)} prompts; "
            f"HTTP POST attempts: {sum(int(call.get('http_post_attempts', 0)) for call in revision_calls)}; "
            f"request time: {sum(float(call.get('duration_seconds', 0)) for call in revision_calls):.3f}s"
        )
        print(f"head width > 1.3x front 3: {wider}")
        print(f"overlay sheet: {overlay}")
        print(f"raw sheet: {raw_sheet}")


if __name__ == "__main__":
    main()
