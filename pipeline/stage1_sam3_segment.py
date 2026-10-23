#!/usr/bin/env python3
"""Stage 1: fal.ai SAM 3 (Segment Anything Model 3) part masking on PAS Seosoyoung canonical front art.

Uses `fal-ai/sam-3/image` with ROI cropping, text prompts, point prompts, and box prompts
to extract high-precision masks on the 1024x1536 canonical canvas (`still.png`).
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageDraw

WORK_DIR = Path(__file__).resolve().parents[1]
SAM3_DIR = WORK_DIR / "sam3_masks"
DEBUG_DIR = WORK_DIR / "debug"
STILL_PATH = WORK_DIR / "source/character/still.png"

SAM3_ENDPOINT = "https://fal.run/fal-ai/sam-3/image"


def image_to_data_uri(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def call_sam3_on_crop(
    fal_key: str,
    full_im: Image.Image,
    crop_box: tuple[int, int, int, int],
    prompt: str,
    points: list[tuple[int, int, int]] | None = None,
    boxes: list[tuple[int, int, int, int]] | None = None,
    upscale: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run SAM 3 on a cropped ROI of `full_im` and map the resulting binary mask back to 1024x1536."""
    x0, y0, x1, y1 = crop_box
    crop = full_im.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    if upscale > 1:
        crop_in = crop.resize((cw * upscale, ch * upscale), Image.Resampling.LANCZOS)
    else:
        crop_in = crop

    bg = Image.new("RGBA", crop_in.size, (238, 240, 245, 255))
    bg.alpha_composite(crop_in)
    rgb_in = bg.convert("RGB")

    payload: dict[str, Any] = {
        "image_url": image_to_data_uri(rgb_in),
        "prompt": prompt,
        "apply_mask": False,
        "return_multiple_masks": True,
        "max_masks": 4,
        "include_scores": True,
        "include_boxes": True,
    }
    if points:
        payload["point_prompts"] = [
            {
                "x": int((px - x0) * upscale),
                "y": int((py - y0) * upscale),
                "label": int(lbl),
                "object_id": 0,
            }
            for (px, py, lbl) in points
        ]
    if boxes:
        payload["box_prompts"] = [
            {
                "x_min": int((bx0 - x0) * upscale),
                "y_min": int((by0 - y0) * upscale),
                "x_max": int((bx1 - x0) * upscale),
                "y_max": int((by1 - y0) * upscale),
                "object_id": 0,
            }
            for (bx0, by0, bx1, by1) in boxes
        ]

    for attempt in range(3):
        resp = requests.post(
            SAM3_ENDPOINT,
            headers={
                "Authorization": f"Key {fal_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        if resp.status_code == 200:
            break
        time.sleep(1.5 * (attempt + 1))
    else:
        raise RuntimeError(f"SAM 3 failed ({resp.status_code}): {resp.text[:300]}")

    data = resp.json()
    masks = data.get("masks", [])
    if not masks:
        raise ValueError("SAM 3 returned an empty mask list")
    scores = data.get("scores") or []

    full_mask = np.zeros((full_im.height, full_im.width), dtype=np.uint8)
    chosen_score = 0.0
    if masks:
        best_idx = 0
        if scores:
            best_idx = int(np.argmax([s if s is not None else -1.0 for s in scores]))
            chosen_score = float(scores[best_idx] or 0.0)
        mask_url = masks[best_idx]["url"]
        download = requests.get(mask_url, timeout=60)
        download.raise_for_status()
        m_bytes = download.content
        m_im = Image.open(io.BytesIO(m_bytes)).convert("L")
        if m_im.size != (cw, ch):
            m_im = m_im.resize((cw, ch), Image.Resampling.BILINEAR)
        full_mask[y0:y1, x0:x1] = np.array(m_im)

    orig_alpha = np.array(full_im)[:, :, 3]
    full_mask = np.minimum(full_mask, orig_alpha)

    meta = {
        "prompt": prompt,
        "crop_box": list(crop_box),
        "upscale": upscale,
        "num_masks_returned": len(masks),
        "score": chosen_score,
        "area_px": int((full_mask > 127).sum()),
    }
    return full_mask, meta


PART_SPECS: list[dict[str, Any]] = [
    {
        "id": "face-skin",
        "crop": (250, 180, 760, 640),
        "prompt": "anime girl face skin",
        "points": [(504, 490, 1), (420, 520, 1), (590, 520, 1), (504, 280, 1), (324, 465, 0), (678, 464, 0), (502, 660, 0)],
        "boxes": [(308, 210, 695, 620)],
        "upscale": 1,
    },
    {
        "id": "ear-left",
        "crop": (260, 380, 400, 550),
        "prompt": "human ear",
        "points": [(324, 465, 1), (315, 440, 1), (360, 465, 0)],
        "boxes": [(290, 410, 366, 518)],
        "upscale": 2,
    },
    {
        "id": "ear-right",
        "crop": (600, 380, 740, 550),
        "prompt": "human ear",
        "points": [(678, 464, 1), (688, 440, 1), (640, 464, 0)],
        "boxes": [(638, 410, 712, 518)],
        "upscale": 2,
    },
    {
        "id": "hairpin",
        "crop": (620, 220, 730, 330),
        "prompt": "blue bunny hair clip ornament",
        "points": [(672, 282, 1), (667, 258, 1)],
        "boxes": [(650, 245, 698, 308)],
        "upscale": 2,
    },
    {
        "id": "tail-left-tie",
        "crop": (220, 480, 395, 610),
        "prompt": "blue spherical hair tie beads",
        "points": [(280, 540, 1), (330, 548, 1), (290, 600, 0)],
        "boxes": [(248, 504, 368, 583)],
        "upscale": 2,
    },
    {
        "id": "tail-right-tie",
        "crop": (605, 480, 775, 610),
        "prompt": "blue spherical hair tie beads",
        "points": [(665, 545, 1), (715, 538, 1), (700, 600, 0)],
        "boxes": [(630, 503, 748, 582)],
        "upscale": 2,
    },
    {
        "id": "eye-left-opening",
        "crop": (320, 360, 470, 490),
        "prompt": "anime eye eyeball",
        "points": [(415, 437, 1), (380, 440, 1), (438, 440, 1)],
        "boxes": [(345, 385, 452, 474)],
        "upscale": 2,
    },
    {
        "id": "eye-right-opening",
        "crop": (535, 360, 685, 490),
        "prompt": "anime eye eyeball",
        "points": [(587, 437, 1), (566, 440, 1), (622, 440, 1)],
        "boxes": [(550, 385, 660, 474)],
        "upscale": 2,
    },
    {
        "id": "eye-left-iris",
        "crop": (320, 360, 470, 490),
        "prompt": "colorful iris of eye",
        "points": [(415, 437, 1), (400, 450, 1), (430, 450, 1), (375, 445, 0)],
        "boxes": [(386, 403, 448, 471)],
        "upscale": 2,
    },
    {
        "id": "eye-right-iris",
        "crop": (535, 360, 685, 490),
        "prompt": "colorful iris of eye",
        "points": [(587, 437, 1), (572, 450, 1), (602, 450, 1), (626, 445, 0)],
        "boxes": [(556, 403, 617, 470)],
        "upscale": 2,
    },
    {
        "id": "bangs-front",
        "crop": (270, 60, 720, 440),
        "prompt": "front bangs hair",
        "points": [(415, 238, 1), (584, 221, 1), (460, 290, 1), (504, 490, 0)],
        "boxes": [(310, 80, 680, 420)],
        "upscale": 1,
    },
    {
        "id": "tail-left-hair",
        "crop": (130, 490, 410, 1150),
        "prompt": "long twin tail ponytail hair",
        "points": [(279, 812, 1), (265, 680, 1), (285, 960, 1), (280, 540, 0), (330, 548, 0)],
        "boxes": [(148, 496, 392, 1130)],
        "upscale": 1,
    },
    {
        "id": "tail-right-hair",
        "crop": (590, 490, 860, 1150),
        "prompt": "long twin tail ponytail hair",
        "points": [(717, 820, 1), (725, 680, 1), (710, 960, 1), (665, 545, 0), (715, 538, 0)],
        "boxes": [(604, 502, 845, 1140)],
        "upscale": 1,
    },
    {
        "id": "necklace",
        "crop": (380, 660, 620, 890),
        "prompt": "gold chain clover pendant necklace",
        "points": [(502, 838, 1), (445, 750, 1), (560, 750, 1)],
        "boxes": [(410, 680, 595, 868)],
        "upscale": 2,
    },
    {
        "id": "neck",
        "crop": (340, 560, 660, 800),
        "prompt": "neck and collarbone skin",
        "points": [(502, 650, 1), (450, 710, 1), (555, 710, 1), (502, 540, 0), (502, 838, 0)],
        "boxes": [(368, 574, 638, 775)],
        "upscale": 1,
    },
    {
        "id": "sweater",
        "crop": (0, 680, 1024, 1536),
        "prompt": "cream white cable knit sweater",
        "points": [(501, 1131, 1), (160, 1100, 1), (850, 1100, 1), (504, 1444, 0)],
        "boxes": [(0, 695, 1024, 1536)],
        "upscale": 1,
    },
    {
        "id": "skirt",
        "crop": (180, 1300, 830, 1536),
        "prompt": "blue plaid pleated skirt",
        "points": [(504, 1444, 1), (360, 1460, 1), (650, 1460, 1)],
        "boxes": [(200, 1318, 810, 1536)],
        "upscale": 1,
    },
    {
        "id": "hair-crown-all",
        "crop": (180, 0, 810, 600),
        "prompt": "upper hair crown and bangs and side locks",
        "points": [(480, 94, 1), (415, 238, 1), (584, 221, 1), (290, 351, 1), (703, 356, 1), (504, 490, 0)],
        "boxes": [(195, 8, 785, 585)],
        "upscale": 1,
    },
    {
        "id": "side-hair-left",
        "crop": (180, 100, 430, 590),
        "prompt": "left side hair lock framing cheek",
        "points": [(290, 351, 1), (260, 260, 1), (345, 500, 1), (324, 465, 0), (415, 437, 0)],
        "boxes": [(198, 120, 402, 575)],
        "upscale": 1,
    },
    {
        "id": "side-hair-right",
        "crop": (580, 100, 800, 590),
        "prompt": "right side hair lock framing cheek",
        "points": [(703, 356, 1), (720, 260, 1), (655, 500, 1), (678, 464, 0), (672, 279, 0)],
        "boxes": [(598, 118, 782, 575)],
        "upscale": 1,
    },
    {
        "id": "eye-left-brow",
        "crop": (320, 320, 470, 390),
        "prompt": "eyebrow line",
        "points": [(398, 344, 1), (365, 348, 1), (432, 348, 1)],
        "boxes": [(336, 336, 454, 372)],
        "upscale": 2,
    },
    {
        "id": "eye-right-brow",
        "crop": (530, 320, 675, 390),
        "prompt": "eyebrow line",
        "points": [(597, 343, 1), (565, 347, 1), (635, 349, 1)],
        "boxes": [(542, 336, 661, 379)],
        "upscale": 2,
    },
    {
        "id": "eye-left-upper-lashes",
        "crop": (320, 365, 470, 470),
        "prompt": "dark upper eyelashes arch",
        "points": [(390, 392, 1), (355, 412, 1), (438, 405, 1), (415, 437, 0)],
        "boxes": [(335, 379, 455, 458)],
        "upscale": 2,
    },
    {
        "id": "eye-right-upper-lashes",
        "crop": (535, 365, 680, 470),
        "prompt": "dark upper eyelashes arch",
        "points": [(611, 391, 1), (565, 405, 1), (648, 412, 1), (587, 437, 0)],
        "boxes": [(547, 376, 669, 457)],
        "upscale": 2,
    },
]


def render_sam3_grid(still_im: Image.Image, results: dict[str, Any]) -> None:
    cols = 6
    rows = 4
    cell_w, cell_h = 220, 260
    grid = Image.new("RGBA", (cols * cell_w + 20, rows * cell_h + 20), (20, 22, 28, 255))
    draw = ImageDraw.Draw(grid)
    still_np = np.array(still_im)

    for idx, spec in enumerate(PART_SPECS[: cols * rows]):
        pid = spec["id"]
        mask_path = SAM3_DIR / f"{pid}_mask.png"
        if not mask_path.exists():
            continue
        m_np = np.array(Image.open(mask_path).convert("L"))
        x0, y0, x1, y1 = spec["crop"]
        crop_rgb = still_np[y0:y1, x0:x1, :3].astype(np.float32)
        crop_m = (m_np[y0:y1, x0:x1] > 127).astype(np.float32)[..., None]

        # Dim outside mask, highlight inside mask with subtle cyan contour tint
        vis = crop_rgb * (0.28 + 0.72 * crop_m)
        vis_im = Image.fromarray(np.clip(vis, 0, 255).astype(np.uint8), "RGB")
        vis_im.thumbnail((cell_w - 16, cell_h - 36), Image.Resampling.LANCZOS)

        gx = (idx % cols) * cell_w + 10
        gy = (idx // cols) * cell_h + 10
        ox = gx + (cell_w - vis_im.width) // 2
        oy = gy + 24 + (cell_h - 30 - vis_im.height) // 2
        grid.paste(vis_im.convert("RGBA"), (ox, oy))
        score = results.get(pid, {}).get("score", 0.0)
        draw.text((gx + 6, gy + 4), f"{pid} ({score:.2f})", fill=(230, 236, 245, 255))

    grid.save(DEBUG_DIR / "sam3_masks_grid.png")


def main() -> int:
    SAM3_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    still_im = Image.open(STILL_PATH).convert("RGBA")

    manifest_out = SAM3_DIR / "sam3_manifest.json"
    results: dict[str, Any] = json.loads(manifest_out.read_text()) if manifest_out.exists() else {}
    todo_specs = [s for s in PART_SPECS if not (SAM3_DIR / f"{s['id']}_mask.png").exists()]

    t0 = time.time()
    if todo_specs:
        fal_key = os.environ["FAL_KEY"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            future_map = {
                pool.submit(
                    call_sam3_on_crop,
                    fal_key,
                    still_im,
                    spec["crop"],
                    spec["prompt"],
                    spec.get("points"),
                    spec.get("boxes"),
                    spec.get("upscale", 1),
                ): spec
                for spec in todo_specs
            }
            for fut in as_completed(future_map):
                spec = future_map[fut]
                pid = spec["id"]
                mask, meta = fut.result()
                out_path = SAM3_DIR / f"{pid}_mask.png"
                Image.fromarray(mask, "L").save(out_path)
                results[pid] = meta
                print(f"[SAM3] {pid:22s} score={meta['score']:.3f} area={meta['area_px']:7d} -> {out_path.name}")

    manifest_out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    render_sam3_grid(still_im, results)
    print(f"[SAM3] Total {len(results)} part masks ready ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
