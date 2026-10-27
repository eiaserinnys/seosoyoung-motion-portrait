#!/usr/bin/env python3
"""Stage 2: deterministic pixel restoration of existing separated artwork.
SAM 3 masks constrain the necklace removal and hairpin crop. Forehead, ear,
neck, eye and hair root fills use local pixel interpolation and alpha geometry.
No external generative inpainting request is made by this stage.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from part_preview import render_before_after_comparison
from part_pixels import dilate_bool, erode_bool, blur_f32, fill_holes_2d, inpaint_horizontal_and_diffuse, despill_green_and_clean_alpha
from neck_art import clear_background

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

WORK_DIR = Path(__file__).resolve().parents[1]
SAM3_DIR = WORK_DIR / "sam3_masks"
INPAINT_DIR = WORK_DIR / "inpainted"
PARTS_DIR = WORK_DIR / "parts"
DEBUG_DIR = WORK_DIR / "debug"

BASE_CHAR_DIR = WORK_DIR / "source/character"
STILL_PATH = BASE_CHAR_DIR / "still.png"


def load_existing_canvas_part(rel_src: str, stage_bounds: dict[str, int]) -> np.ndarray:
    im = Image.open(BASE_CHAR_DIR / rel_src).convert("RGBA")
    w, h = stage_bounds["width"], stage_bounds["height"]
    x, y = stage_bounds["x"], stage_bounds["y"]
    im_r = im.resize((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (1024, 1536), (0, 0, 0, 0))
    canvas.paste(im_r, (x, y))
    return np.array(canvas)


def load_sam3_mask(part_id: str) -> np.ndarray:
    p = SAM3_DIR / f"{part_id}_mask.png"
    return np.array(Image.open(p).convert("L"))


def build_clean_face_skin(still_np: np.ndarray, existing_face: np.ndarray) -> np.ndarray:
    """Rebuild `face-skin` so the forehead V-seam, temple steps at y=285..325, and right-cheek speck are 100% gone."""
    fx0, fy0, fx1, fy1 = 272, 220, 752, 640
    ex_crop = existing_face[fy0:fy1, fx0:fx1].copy()
    ex_rgb = ex_crop[:, :, :3].astype(np.float32)
    ex_a = ex_crop[:, :, 3].astype(np.float32)

    h, w = fy1 - fy0, fx1 - fx0
    yy, xx = np.indices((h, w))
    cy = yy + fy0
    cx = xx + fx0

    clean_a = ex_a.copy()
    # Remove right-cheek speck near (680..700, 515..540)
    speck_mask = (cx >= 674) & (cy >= 515) & (cy <= 542) & (clean_a < 220)
    clean_a = np.where(speck_mask | ((cx >= 683) & (cy >= 518) & (cy <= 538)), 0.0, clean_a)

    # Smooth the temple notch step (which occurs around cy=280..325) by fitting a quadratic curve between cy=245 and cy=360
    left_x = np.zeros(h, dtype=np.float32)
    right_x = np.zeros(h, dtype=np.float32)
    valid_rows = []
    for y in range(h):
        xs = np.where(clean_a[y] > 128)[0]
        if len(xs) >= 10:
            left_x[y] = float(xs[0])
            right_x[y] = float(xs[-1])
            valid_rows.append(y)

    anchor_ys = [y for y in valid_rows if (232 <= (y + fy0) <= 265) or (345 <= (y + fy0) <= 385)]
    if len(anchor_ys) >= 6:
        poly_l = np.poly1d(np.polyfit(anchor_ys, [left_x[y] for y in anchor_ys], 2))
        poly_r = np.poly1d(np.polyfit(anchor_ys, [right_x[y] for y in anchor_ys], 2))
        for y in range(250 - fy0, 365 - fy0):
            lx = float(poly_l(y))
            rx = float(poly_r(y))
            row_x = np.arange(w, dtype=np.float32)
            arch_a = np.clip(np.minimum(row_x - lx + 1.0, rx - row_x + 1.0) * 128.0, 0.0, 255.0)
            clean_a[y] = arch_a

    # Clean interior RGB: remove the V-shaped hair line and temple outline above cy=375
    interior_mask = erode_bool(clean_a > 200, 10)
    ref_band = ex_rgb[368 - fy0 : 392 - fy0, :, :]
    ref_valid = interior_mask[368 - fy0 : 392 - fy0, :]
    col_skin = np.zeros((w, 3), dtype=np.float32)
    default_skin = np.array([255.0, 232.0, 217.0], dtype=np.float32)
    for x in range(w):
        vs = ref_band[ref_valid[:, x], x]
        if len(vs) > 0:
            col_skin[x] = vs.mean(axis=0)
        else:
            col_skin[x] = default_skin
    col_skin = blur_f32(col_skin[None, :, :], 18.0)[0]

    t_y = np.clip((372.0 - cy) / 148.0, 0.0, 1.0)[..., None]
    top_tint = np.array([252.0, 223.0, 206.0], dtype=np.float32)
    forehead_field = col_skin[None, :, :] * (1.0 - 0.45 * t_y) + top_tint * (0.45 * t_y)

    replace_zone = ((cy < 364) | ((cy < 430) & (~interior_mask))).astype(np.float32)
    replace_w = blur_f32(replace_zone, 6.0)[..., None]
    clean_rgb = ex_rgb * (1.0 - replace_w) + forehead_field * replace_w

    out_canvas = np.zeros((1536, 1024, 4), dtype=np.uint8)
    out_canvas[fy0:fy1, fx0:fx1, :3] = np.clip(clean_rgb, 0, 255).astype(np.uint8)
    out_canvas[fy0:fy1, fx0:fx1, 3] = np.clip(clean_a, 0, 255).astype(np.uint8)
    return out_canvas


def build_extended_ear(existing_ear: np.ndarray, side: str) -> np.ndarray:
    """Replace the inner edge from an eroded core with a smooth 16px skin root."""
    out = despill_green_and_clean_alpha(existing_ear, None, lock_visible=False)
    core = erode_bool(existing_ear[:, :, 3] > 235, 2)
    rgb = existing_ear[:, :, :3].astype(np.float32)
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    skin = core & (luma > 175)
    rows = np.flatnonzero(skin.any(axis=1))
    if not len(rows):
        raise ValueError(f"No opaque interior skin samples for {side} ear")
    profile = np.array([rgb[y, skin[y]].mean(axis=0) for y in rows])
    profile = np.stack([np.interp(np.arange(out.shape[0]), rows, profile[:, c]) for c in range(3)], axis=1)
    profile = blur_f32(profile[:, None, :], 10.0)[:, 0, :]
    direction = 1 if side == "left" else -1
    edge_rows = np.flatnonzero(core.any(axis=1))
    edges = np.array([np.flatnonzero(core[y])[-1 if side == "left" else 0] for y in edge_rows])
    edge_profile = np.interp(np.arange(out.shape[0]), edge_rows, edges)
    # Coordinates require a float kernel; the image blur clips values to 255.
    offsets = np.arange(-8, 9, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / 2.5) ** 2)
    kernel /= kernel.sum()
    edge_profile = np.convolve(np.pad(edge_profile, (8, 8), mode="edge"), kernel, mode="valid")
    for y in edge_rows:
        inner_x = int(round(edge_profile[y]))
        for dx in range(17):
            tx = inner_x + direction * dx
            if 0 <= tx < out.shape[1]:
                out[y, tx, :3] = profile[y].astype(np.uint8)
                # Feather the outer six pixels while retaining the original cartilage.
                alpha = int(255 * np.clip((16 - dx) / 6.0, 0, 1))
                out[y, tx, 3] = max(int(out[y, tx, 3]), alpha)
    cartilage = erode_bool(existing_ear[:, :, 3] > 235, 4) & (luma <= 175)
    out[cartilage] = existing_ear[cartilage]
    return out


def build_clean_neck(existing_neck: np.ndarray, existing_shirt: np.ndarray) -> np.ndarray:
    """Remove ONLY the narrow gold necklace chain from `neck` (preserving 100% of collarbone cel-shading) and extend upper neck cleanly."""
    existing_neck = existing_neck.copy()
    shirt_region = existing_shirt[580:726, 345:656]
    neck_region = existing_neck[580:726, 345:656]
    rgb = shirt_region[:, :, :3].astype(np.float32)
    skin = (rgb[:, :, 0] > rgb[:, :, 1] + 8) & (rgb[:, :, 1] > rgb[:, :, 2] + 3) & (rgb[:, :, 0] > 100)
    merge = skin & (shirt_region[:, :, 3] > neck_region[:, :, 3])
    neck_region[merge] = shirt_region[merge]
    nx0, ny0, nx1, ny1 = 320, 512, 688, 832
    ex_crop = existing_neck[ny0:ny1, nx0:nx1].copy()
    ex_rgb = ex_crop[:, :, :3].astype(np.float32)
    ex_alpha = ex_crop[:, :, 3].astype(np.float32) / 255.0

    # Use strictly the SAM 3 necklace mask dilated by 3px (never touch collarbone shadows!)
    sam_necklace = load_sam3_mask("necklace")[ny0:ny1, nx0:nx1] > 28
    chain_mask = dilate_bool(sam_necklace, 3)

    valid_skin = (ex_alpha > 0.85) & (~chain_mask)
    inpainted_rgb = inpaint_horizontal_and_diffuse(ex_rgb, valid_skin, passes=8)

    blend_w = blur_f32(chain_mask.astype(np.float32), 1.0)[..., None]
    clean_rgb = ex_rgb * (1.0 - blend_w) + inpainted_rgb * blend_w

    # Extend upper neck from y_canvas=577 up to y_canvas=536 using row 580's clean under-chin shadow color
    ref_row = 580 - ny0
    top_row_rgb = clean_rgb[ref_row].copy()
    top_row_a = ex_alpha[ref_row].copy()
    for cy in range(536, 580):
        ry = cy - ny0
        t = max(0.0, (580.0 - cy) / 44.0)
        fade = np.clip(1.0 - t**1.8, 0.0, 1.0)
        clean_rgb[ry] = top_row_rgb
        ex_alpha[ry] = np.maximum(ex_alpha[ry], top_row_a * fade)

    out_canvas = np.zeros((1536, 1024, 4), dtype=np.uint8)
    out_canvas[ny0:ny1, nx0:nx1, :3] = np.clip(clean_rgb, 0, 255).astype(np.uint8)
    out_canvas[ny0:ny1, nx0:nx1, 3] = np.clip(ex_alpha * 255.0, 0, 255).astype(np.uint8)
    return out_canvas


def build_hole_free_sclera_and_iris(existing_map: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Create smooth almond hole-free `eye-left/right-sclera` and hole-free `eye-left/right-iris`."""
    out: dict[str, np.ndarray] = {}

    for side in ("left", "right"):
        ex_sclera = existing_map[f"eye-{side}-sclera"].copy()
        ex_iris = existing_map[f"eye-{side}-iris"]
        ex_pupil = existing_map[f"eye-{side}-pupil"]
        ex_high = existing_map[f"eye-{side}-highlight"]

        # Remove the isolated dust speck at y < 403 in ex_sclera before forming the almond union
        ex_sclera[:403, :, 3] = 0

        raw_union = (ex_sclera[:, :, 3] > 24) | (ex_iris[:, :, 3] > 24) | (ex_pupil[:, :, 3] > 24) | (ex_high[:, :, 3] > 24)
        filled_union = fill_holes_2d(raw_union)
        closed_union = erode_bool(dilate_bool(filled_union, 4), 4)
        smooth_a = blur_f32(closed_union.astype(np.float32), 1.6)
        sclera_alpha = np.clip((smooth_a - 0.25) / 0.55 * 255.0, 0, 255).astype(np.uint8)
        sclera_alpha = np.maximum(sclera_alpha, ex_sclera[:, :, 3])

        ys, xs = np.where(sclera_alpha > 8)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())

        sclera_canvas = np.zeros((1536, 1024, 4), dtype=np.uint8)
        yy, _ = np.indices((y1 - y0 + 1, x1 - x0 + 1))
        t_y = np.clip(yy / float(max(1, y1 - y0)), 0.0, 1.0)[..., None]

        top_col = np.array([208.0, 195.0, 204.0], dtype=np.float32)
        mid_col = np.array([247.0, 240.0, 241.0], dtype=np.float32)
        bot_col = np.array([254.0, 249.0, 248.0], dtype=np.float32)
        grad_rgb = np.where(
            t_y < 0.36,
            top_col * (1.0 - t_y / 0.36) + mid_col * (t_y / 0.36),
            mid_col * (1.0 - (t_y - 0.36) / 0.64) + bot_col * ((t_y - 0.36) / 0.64),
        )

        ex_sc_crop = ex_sclera[y0 : y1 + 1, x0 : x1 + 1]
        ex_w = (ex_sc_crop[:, :, 3:4].astype(np.float32) / 255.0) ** 1.4
        merged_sc_rgb = grad_rgb * (1.0 - ex_w) + ex_sc_crop[:, :, :3].astype(np.float32) * ex_w

        sclera_canvas[y0 : y1 + 1, x0 : x1 + 1, :3] = np.clip(merged_sc_rgb, 0, 255).astype(np.uint8)
        sclera_canvas[y0 : y1 + 1, x0 : x1 + 1, 3] = sclera_alpha[y0 : y1 + 1, x0 : x1 + 1]
        out[f"eye-{side}-sclera"] = sclera_canvas

        iris_canvas = ex_iris.copy()
        iris_outer = np.maximum(ex_iris[:, :, 3], np.maximum(ex_pupil[:, :, 3], ex_high[:, :, 3]))
        iris_filled_bin = fill_holes_2d(iris_outer > 90)
        iris_alpha = np.maximum(ex_iris[:, :, 3], (iris_filled_bin.astype(np.uint8) * 255))

        iys, ixs = np.where(iris_alpha > 8)
        iy0, iy1 = int(iys.min()), int(iys.max())
        ix0, ix1 = int(ixs.min()), int(ixs.max())

        icrop_rgb = iris_canvas[iy0 : iy1 + 1, ix0 : ix1 + 1, :3].astype(np.float32)
        icrop_a = ex_iris[iy0 : iy1 + 1, ix0 : ix1 + 1, 3]
        luma_i = 0.299 * icrop_rgb[:, :, 0] + 0.587 * icrop_rgb[:, :, 1] + 0.114 * icrop_rgb[:, :, 2]
        valid_iris = (icrop_a > 210) & (luma_i < 228.0)
        if valid_iris.any():
            inpainted_iris_rgb = inpaint_horizontal_and_diffuse(icrop_rgb, valid_iris, passes=16)
            w_keep = np.clip((icrop_a.astype(np.float32) - 140.0) / 95.0, 0.0, 1.0)[..., None]
            final_iris_rgb = inpainted_iris_rgb * (1.0 - w_keep) + icrop_rgb * w_keep
            iris_canvas[iy0 : iy1 + 1, ix0 : ix1 + 1, :3] = np.clip(final_iris_rgb, 0, 255).astype(np.uint8)
            iris_canvas[iy0 : iy1 + 1, ix0 : ix1 + 1, 3] = iris_alpha[iy0 : iy1 + 1, ix0 : ix1 + 1]

        out[f"eye-{side}-iris"] = iris_canvas

    return out


def build_clean_hair_and_tails(
    still_np: np.ndarray,
    existing_map: dict[str, np.ndarray],
    face_skin: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build clean, hole-free `hair-top`, `bangs-left/right`, `side-hair-left/right`, `tail-left/right-hair`, and ornaments."""
    out: dict[str, np.ndarray] = {}

    sam_pin = load_sam3_mask("hairpin")
    pin_part = existing_map["hairpin"].copy()
    pin_part[:, :, 3] = np.minimum(pin_part[:, :, 3], dilate_bool(sam_pin > 40, 2).astype(np.uint8) * 255)
    out["hairpin"] = pin_part

    # Blue bead hair-ties: keep 100% of the blue bead circles from `existing_map`, remove only bottom brown hair spikes (y > 565 and R > B)
    for side in ("left", "right"):
        tid = f"tail-{side}-tie"
        tie_part = existing_map[tid].copy()
        r = tie_part[:, :, 0].astype(np.int16)
        b = tie_part[:, :, 2].astype(np.int16)
        yy, _ = np.indices((1536, 1024))
        brown_spike = (yy > 562) & (r >= b - 8)
        tie_part[:, :, 3] = np.where(brown_spike, 0, tie_part[:, :, 3])
        out[tid] = tie_part

    # Twin-tail hair: fill the circular cutout under the blue bead tie smoothly
    for side, root_cx, root_cy, (tx0, ty0, tx1, ty1) in [
        ("left", 308, 544, (144, 488, 400, 808)),
        ("right", 688, 542, (600, 488, 856, 808)),
    ]:
        lid = f"tail-{side}-hair"
        base_tail = despill_green_and_clean_alpha(existing_map[lid], still_np, lock_visible=True)
        crop_rgba = base_tail[ty0:ty1, tx0:tx1].copy()

        yy, xx = np.indices((ty1 - ty0, tx1 - tx0))
        cy = yy + ty0
        cx = xx + tx0

        stray_crumbs = (cy < 528) | ((cy < 562) & (np.abs(cx - root_cx) > 46))
        crop_rgba[:, :, 3] = np.where(stray_crumbs, 0, crop_rgba[:, :, 3])

        # Extend only the lower attachment into the existing hair silhouette.
        tie_core = erode_bool(out[f"tail-{side}-tie"][:, :, 3] > 180, 1)[ty0:ty1, tx0:tx1]
        lower_core = tie_core.copy()
        for shift in range(1, 7):
            lower_core[shift:, :] |= tie_core[:-shift, :] & (cy[shift:, :] >= 552)
        hair_attachment = (crop_rgba[:, :, 3] > 120) & (cy >= 552) & (cy <= 605)
        tie_core = fill_holes_2d(lower_core | hair_attachment)
        arch_y = 548.0 + 0.018 * (cx - root_cx) ** 2
        root_mask = tie_core & (cy >= arch_y) & (cy <= 605)
        root_alpha = blur_f32(root_mask.astype(np.float32), 1.5)
        root_alpha *= tie_core & (cy >= 542)
        root_alpha[erode_bool(root_mask, 1) & (cy >= 552)] = 1.0
        sample_band = base_tail[578:606, tx0:tx1, :3].astype(np.float32)
        sample_a = base_tail[578:606, tx0:tx1, 3] > 180
        columns = np.flatnonzero(sample_a.any(axis=0))
        if not len(columns):
            raise ValueError(f"No opaque hair samples for {side} tail")
        col_hair = np.array([sample_band[np.flatnonzero(sample_a[:, x])[0], x] for x in columns])
        col_hair = np.stack([np.interp(np.arange(tx1 - tx0), columns, col_hair[:, c]) for c in range(3)], axis=1)
        root_rgb = blur_f32(np.broadcast_to(col_hair[None, :, :], crop_rgba[:, :, :3].shape), 2.5)
        weight = crop_rgba[:, :, 3:4].astype(np.float32) / 255.0
        under = root_alpha[..., None] * (1 - weight)
        merged_alpha = weight + under
        merged_rgb = (crop_rgba[:, :, :3] * weight + root_rgb * under) / np.maximum(merged_alpha, 1e-6)
        crop_rgba[:, :, :3] = np.clip(merged_rgb, 0, 255).astype(np.uint8)
        crop_rgba[:, :, 3] = np.clip(merged_alpha[:, :, 0] * 255, 0, 255).astype(np.uint8)

        base_tail[ty0:ty1, tx0:tx1] = crop_rgba
        out[lid] = base_tail

    # `hair-top` (Full Crown Under-Hair)
    hx0, hy0, hx1, hy1 = 192, 0, 800, 420
    base_htop = despill_green_and_clean_alpha(existing_map["hair-top"], still_np, lock_visible=True)
    back_center = despill_green_and_clean_alpha(existing_map["back-hair-center"], None, lock_visible=False)

    htop_crop = base_htop[hy0:hy1, hx0:hx1].copy()
    back_crop = back_center[hy0:hy1, hx0:hx1].copy()

    yy, xx = np.indices((hy1 - hy0, hx1 - hx0))
    cy = yy + hy0
    cx = xx + hx0

    ell_d = np.sqrt(((cx - 495.0) / 235.0) ** 2 + ((cy - 145.0) / 115.0) ** 2)
    cushion_a = np.clip((1.0 - ell_d) * 3.2, 0.0, 1.0) * 255.0
    cushion_a = np.minimum(cushion_a, back_crop[:, :, 3].astype(np.float32))
    hairline = 232.0 - 28.0 * np.clip((cx - 495.0) / 235.0, -1, 1) ** 2
    cushion_a *= np.clip((hairline - cy) / 2.0, 0, 1)
    cushion_a[face_skin[hy0:hy1, hx0:hx1, 3] > 240] = 0

    ex_a = htop_crop[:, :, 3:4].astype(np.float32) / 255.0
    merged_htop_rgb = htop_crop[:, :, :3].astype(np.float32) * ex_a + back_crop[:, :, :3].astype(np.float32) * (1.0 - ex_a)
    htop_crop[:, :, :3] = np.clip(merged_htop_rgb, 0, 255).astype(np.uint8)
    htop_crop[:, :, 3] = np.clip(np.maximum(htop_crop[:, :, 3].astype(np.float32), cushion_a), 0, 255).astype(np.uint8)
    base_htop[hy0:hy1, hx0:hx1] = htop_crop
    out["hair-top"] = base_htop

    for hid in ("bangs-left", "bangs-right", "side-hair-left", "side-hair-right", "back-hair-left", "back-hair-right", "back-hair-center"):
        out[hid] = despill_green_and_clean_alpha(existing_map[hid], still_np, lock_visible=(hid != "back-hair-center"))

    bl = out["bangs-left"]
    br = out["bangs-right"]
    for y in range(88, 215):
        for dx in range(1, 14):
            fade = np.clip(1.0 - dx / 14.0, 0.0, 1.0)
            tx_r = 494 + dx
            if bl[y, 494, 3] > 180 and br[y, tx_r, 3] > 180:
                bl[y, tx_r, :3] = still_np[y, tx_r, :3]
                bl[y, tx_r, 3] = max(bl[y, tx_r, 3], int(220 * fade))
            tx_l = 495 - dx
            if br[y, 495, 3] > 180 and bl[y, tx_l, 3] > 180:
                br[y, tx_l, :3] = still_np[y, tx_l, :3]
                br[y, tx_l, 3] = max(br[y, tx_l, 3], int(220 * fade))

    return out




def main() -> int:
    INPAINT_DIR.mkdir(parents=True, exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    still_im = Image.open(STILL_PATH).convert("RGBA")
    still_np = np.array(still_im)

    manifest = json.loads((BASE_CHAR_DIR / "manifest.json").read_text())
    clothing = json.loads((BASE_CHAR_DIR / "clothing.json").read_text())

    existing_map: dict[str, np.ndarray] = {}
    meta_map: dict[str, dict[str, Any]] = {}
    for leaf in manifest["leaves"]:
        lid = leaf["id"]
        existing_map[lid] = load_existing_canvas_part(leaf["src"], leaf["stageBounds"])
        meta_map[lid] = leaf
    for part in clothing["parts"]:
        pid = part["id"]
        existing_map[pid] = load_existing_canvas_part(part["src"], part["stageBounds"])
        meta_map[pid] = part
    for expr in manifest.get("expressions", []):
        eid = expr["id"]
        existing_map[eid] = load_existing_canvas_part(expr["src"], expr["stageBounds"])
        meta_map[eid] = expr

    t0 = time.time()
    rebuilt_map: dict[str, np.ndarray] = dict(existing_map)

    rebuilt_map["face-skin"] = build_clean_face_skin(still_np, existing_map["face-skin"])
    rebuilt_map["ear-left"] = build_extended_ear(existing_map["ear-left"], "left")
    rebuilt_map["ear-right"] = build_extended_ear(existing_map["ear-right"], "right")
    rebuilt_map["neck"] = clear_background(build_clean_neck(existing_map["neck"], existing_map["upper-body-short-sleeve"]), still_np)

    eye_rebuilt = build_hole_free_sclera_and_iris(existing_map)
    rebuilt_map.update(eye_rebuilt)

    hair_rebuilt = build_clean_hair_and_tails(still_np, existing_map, rebuilt_map["face-skin"])
    rebuilt_map.update(hair_rebuilt)

    for pid in (
        "eye-left-highlight",
        "eye-right-highlight",
        "eye-left-pupil",
        "eye-right-pupil",
        "eye-left-upper-lashes",
        "eye-right-upper-lashes",
        "eye-left-lower-line",
        "eye-right-lower-line",
        "eye-left-brow",
        "eye-right-brow",
        "nose-detail",
        "mouth-detail",
        "upper-body-short-sleeve",
        "necklace",
        "sweater",
        "skirt",
        "closed-eyes-v1",
        "mouth-open-v1",
        "mouth-round-v1",
    ):
        if pid in rebuilt_map:
            rebuilt_map[pid] = despill_green_and_clean_alpha(
                rebuilt_map[pid],
                still_np,
                lock_visible=(pid not in ("closed-eyes-v1", "mouth-open-v1", "mouth-round-v1", "upper-body-short-sleeve")),
            )

    shirt = rebuilt_map["upper-body-short-sleeve"].copy()
    fade = np.clip((np.arange(1536, dtype=np.float32) - 680.0) / 12.0, 0.0, 1.0)
    shirt[:, :, 3] = (shirt[:, :, 3].astype(np.float32) * fade[:, None]).astype(np.uint8)
    rebuilt_map["upper-body-short-sleeve"] = shirt

    parts_index: list[dict[str, Any]] = []
    for pid, canvas_rgba in rebuilt_map.items():
        ys, xs = np.where(canvas_rgba[:, :, 3] > 2)
        if len(xs) == 0:
            continue
        pad = 6
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(1024, int(xs.max()) + 1 + pad)
        y1 = min(1536, int(ys.max()) + 1 + pad)
        crop_im = Image.fromarray(canvas_rgba[y0:y1, x0:x1], "RGBA")
        out_file = PARTS_DIR / f"{pid}.png"
        crop_im.save(out_file)
        parts_index.append(
            {
                "id": pid,
                "label": meta_map.get(pid, {}).get("label", pid),
                "file": f"parts/{pid}.png",
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area_px": int((canvas_rgba[:, :, 3] > 16).sum()),
            }
        )

    (PARTS_DIR / "parts_index.json").write_text(json.dumps(parts_index, indent=2, ensure_ascii=False))
    render_before_after_comparison(existing_map, rebuilt_map, DEBUG_DIR)
    print(f"[Stage2] Saved {len(parts_index)} rebuilt parts and comparison grid in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
