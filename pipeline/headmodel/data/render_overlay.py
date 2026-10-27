#!/usr/bin/env python3
"""Render registered head-model masks and optional landmarks into a sheet."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent.parent
FRAME_DIR = WORKSPACE_ROOT / ".local/artifacts/20261025-ponytail-seedance-reg/frames"
WIDTH = 720
HEIGHT = 720
CELL = 360


def load_registered_frame(
    frame_number: int,
    key_row: dict[str, object] | None,
) -> tuple[np.ndarray, bool]:
    path = FRAME_DIR / f"frame_{frame_number:04d}.png"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing source video frame: {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    normalization = key_row.get("normalization") if key_row else None
    if normalization:
        matrix = np.asarray(normalization, dtype=np.float32)
        if matrix.shape != (2, 3):
            raise ValueError(f"Frame {frame_number} normalization must be 2x3")
        rgb = cv2.warpAffine(
            rgb,
            matrix,
            (WIDTH, HEIGHT),
            flags=cv2.INTER_CUBIC,
            borderValue=(255, 255, 255),
        )
        return rgb, True
    return rgb, False


def render_overlay_sheet(
    registration: dict[str, object],
    frame_rows: list[dict[str, object]],
    landmarks: dict[int, dict[str, object]] | None = None,
) -> Path:
    key_rows = {int(row["sourceFrame"]): row for row in registration["keys"]}
    available = [row for row in frame_rows if (DATA_DIR / f"frame_{int(row['frame']):04d}").is_dir()]
    columns = 6
    sheet = np.full((int(np.ceil(len(available) / columns)) * CELL, columns * CELL, 3), 248, dtype=np.uint8)
    for index, row in enumerate(available):
        frame_number = int(row["frame"])
        rgb, _ = load_registered_frame(frame_number, key_rows.get(frame_number))
        canvas = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (CELL, CELL), interpolation=cv2.INTER_AREA)
        frame_dir = DATA_DIR / f"frame_{frame_number:04d}"
        for side in ("ear-left", "ear-right"):
            ear = cv2.imread(str(frame_dir / f"{side}.png"), cv2.IMREAD_GRAYSCALE)
            if ear is not None and int(cv2.countNonZero(ear)):
                resized = cv2.resize(ear, (CELL, CELL), interpolation=cv2.INTER_NEAREST) > 0
                canvas[resized] = (0, 255, 255)
        for name, color in (("head", (0, 255, 0)), ("skin", (255, 0, 0))):
            mask = cv2.imread(str(frame_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                resized = cv2.resize(mask, (CELL, CELL), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas, contours, -1, color, 2, cv2.LINE_AA)
        if landmarks and frame_number in landmarks:
            for x, y, score in landmarks[frame_number].get("keypoints", []):
                if float(score) >= 0.0:
                    cv2.circle(canvas, (int(round(float(x) / 2)), int(round(float(y) / 2))), 3, (0, 0, 255), -1, cv2.LINE_AA)
        yaw, pitch = float(row["pose"][0]), float(row["pose"][1])
        cv2.putText(canvas, f"Frame {frame_number}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"yaw {yaw:.1f}  pitch {pitch:.1f}", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        r, c = divmod(index, columns)
        sheet[r * CELL : (r + 1) * CELL, c * CELL : (c + 1) * CELL] = canvas
    target = DATA_DIR / "sheets/masks-overlay.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), sheet):
        raise RuntimeError(f"Could not write overlay sheet: {target}")
    return target


def main() -> None:
    from prepare_video_inputs import build_frame_index, read_registration

    registration = read_registration()
    frame_rows = [row for row in build_frame_index(registration) if row["role"] == "key"]
    landmark_path = DATA_DIR / "landmarks.json"
    landmarks = None
    if landmark_path.exists():
        payload = json.loads(landmark_path.read_text(encoding="utf-8"))
        landmarks = {int(row["frame"]): row for row in payload["frames"]}
    print(render_overlay_sheet(registration, frame_rows, landmarks))


if __name__ == "__main__":
    main()
