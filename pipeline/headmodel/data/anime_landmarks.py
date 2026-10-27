#!/usr/bin/env python3
"""Run the optional CPU anime face detector on the nine registered key frames."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from anime_face_detector import create_detector
from importlib.metadata import version


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent.parent
REGISTRATION_PATH = REPO_ROOT / "source/viewer/ponytail-seedance-registration.json"
FRAME_DIR = WORKSPACE_ROOT / ".local/artifacts/20261025-ponytail-seedance-reg/frames"
OUTPUT_PATH = Path(__file__).resolve().parent / "landmarks.json"
WIDTH = 720
HEIGHT = 720
SCORE_THRESHOLD = 0.5
YAW_LIMIT_DEGREES = 35.0


def load_registered_frame(frame_number: int, key_row: dict[str, object]) -> np.ndarray:
    path = FRAME_DIR / f"frame_{frame_number:04d}.png"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing source video frame: {path}")
    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    normalization = key_row.get("normalization")
    if normalization:
        matrix = np.asarray(normalization, dtype=np.float32)
        if matrix.shape != (2, 3):
            raise ValueError(f"Frame {frame_number} normalization must be 2x3")
        image = cv2.warpAffine(
            image,
            matrix,
            (WIDTH, HEIGHT),
            flags=cv2.INTER_CUBIC,
            borderValue=(255, 255, 255),
        )
    return image


def main() -> None:
    registration = json.loads(REGISTRATION_PATH.read_text(encoding="utf-8"))
    key_rows = sorted(registration["keys"], key=lambda row: int(row["sourceFrame"]))
    torch.set_num_threads(2)
    detector = create_detector("yolov3", device="cpu")
    frames = []
    for key_row in key_rows:
        frame_number = int(key_row["sourceFrame"])
        pose = [float(value) for value in registration["video"][frame_number - 1]["pose"]]
        started = time.perf_counter()
        predictions = detector(load_registered_frame(frame_number, key_row))
        selected = max(predictions, key=lambda item: float(item["bbox"][4])) if predictions else None
        if selected is None:
            score = 0.0
            bbox: list[float] = []
            keypoints: list[list[float]] = []
            reasons = ["no_face_detected"]
        else:
            bbox_values = np.asarray(selected["bbox"], dtype=np.float64).tolist()
            score = float(bbox_values[4]) if len(bbox_values) > 4 else 0.0
            bbox = [float(value) for value in bbox_values[:4]]
            keypoint_array = np.asarray(selected["keypoints"], dtype=np.float64)
            if keypoint_array.shape != (28, 3):
                raise ValueError(f"Frame {frame_number} detector returned {keypoint_array.shape}, expected (28, 3)")
            keypoints = [[float(x), float(y), float(confidence)] for x, y, confidence in keypoint_array]
            reasons = []
            if score < SCORE_THRESHOLD:
                reasons.append("score_below_0.5")
        yaw = pose[0]
        if abs(yaw) > YAW_LIMIT_DEGREES:
            reasons.append("absolute_yaw_over_35_degrees")
        frames.append(
            {
                "frame": frame_number,
                "key_name": key_row["name"],
                "pose": pose,
                "bbox": bbox,
                "score": score,
                "score_definition": "selected face detector bbox confidence (bbox[4])",
                "keypoints": keypoints,
                "usable": selected is not None and score >= SCORE_THRESHOLD and abs(yaw) <= YAW_LIMIT_DEGREES,
                "reasons": reasons,
                "duration_seconds": round(time.perf_counter() - started, 3),
            }
        )
        print(f"frame {frame_number}: score={score:.3f}, usable={frames[-1]['usable']}", flush=True)

    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "model": f"anime-face-detector {version('anime-face-detector')} / yolov3 + hrnetv2",
                "device": "cpu",
                "landmark_count": 28,
                "coordinate_space": "registered 720x720 frame",
                "score_threshold": SCORE_THRESHOLD,
                "usable_yaw_limit_degrees": YAW_LIMIT_DEGREES,
                "frames": frames,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
