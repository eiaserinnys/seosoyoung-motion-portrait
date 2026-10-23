"""Pure pixel operations shared by the part restoration pipeline."""
import numpy as np
from PIL import Image, ImageFilter

def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    im = Image.fromarray((mask.astype(np.uint8) * 255), "L")
    im = im.filter(ImageFilter.MaxFilter(size=2 * radius + 1))
    return np.array(im) > 127


def erode_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    im = Image.fromarray((mask.astype(np.uint8) * 255), "L")
    im = im.filter(ImageFilter.MinFilter(size=2 * radius + 1))
    return np.array(im) > 127


def blur_f32(arr: np.ndarray, sigma: float) -> np.ndarray:
    if arr.ndim == 2:
        scale = 255.0 if arr.max() <= 1.01 else 1.0
        im = Image.fromarray(np.clip(arr * scale, 0, 255).astype(np.uint8), "L")
        im = im.filter(ImageFilter.GaussianBlur(radius=sigma))
        res = np.array(im).astype(np.float32)
        return res / scale
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    im = im.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.array(im).astype(np.float32)


def fill_holes_2d(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = out.shape
    row_filled = np.zeros_like(out)
    col_filled = np.zeros_like(out)
    for y in range(h):
        xs = np.where(out[y])[0]
        if len(xs) >= 2:
            row_filled[y, xs[0] : xs[-1] + 1] = True
    for x in range(w):
        ys = np.where(out[:, x])[0]
        if len(ys) >= 2:
            col_filled[ys[0] : ys[-1] + 1, x] = True
    return out | (row_filled & col_filled)


def inpaint_horizontal_and_diffuse(rgb: np.ndarray, valid_mask: np.ndarray, passes: int = 12) -> np.ndarray:
    """Inpaint invalid pixels by horizontal scanline interpolation from valid neighbors, then lightly smooth inside the mask."""
    out = rgb.astype(np.float32).copy()
    h, w, _ = out.shape
    for y in range(h):
        v_xs = np.where(valid_mask[y])[0]
        if len(v_xs) == 0:
            continue
        inv_xs = np.where(~valid_mask[y])[0]
        if len(inv_xs) == 0:
            continue
        for c in range(3):
            out[y, inv_xs, c] = np.interp(inv_xs, v_xs, out[y, v_xs, c])
    for sigma in [2.5, 1.2]:
        for _ in range(passes // 2):
            smoothed = blur_f32(out, sigma)
            out = np.where(valid_mask[..., None], rgb.astype(np.float32), smoothed)
    return out


def despill_green_and_clean_alpha(rgba: np.ndarray, still_np: np.ndarray | None = None, lock_visible: bool = False) -> np.ndarray:
    """Remove green chroma-key spill on hair/part fringes."""
    out = rgba.copy().astype(np.float32)
    r, g, b, a = out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]

    rb_max = np.maximum(r, b)
    green_excess = np.maximum(0.0, g - rb_max * 0.95)
    mask_spill = (green_excess > 1.5) & (a > 0)
    out[:, :, 1] = np.where(mask_spill, 0.62 * r + 0.38 * b, g)
    out[:, :, 0] = np.where(mask_spill, np.minimum(255.0, r + green_excess * 0.18), r)
    out[:, :, 3] = np.where(green_excess > 16.0, a * 0.4, a)

    luma = 0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2]
    halo = (a > 0) & (a < 90) & (luma > 215)
    out[:, :, 3] = np.where(halo, a * 0.2, out[:, :, 3])

    if still_np is not None and lock_visible:
        s_rgb = still_np[:, :, :3].astype(np.float32)
        s_a = still_np[:, :, 3].astype(np.float32) / 255.0
        diff = np.abs(out[:, :, :3] - s_rgb).mean(axis=2)
        match_w = np.clip((36.0 - diff) / 24.0, 0.0, 1.0) * s_a * (out[:, :, 3] > 64).astype(np.float32)
        out[:, :, :3] = out[:, :, :3] * (1.0 - match_w[..., None]) + s_rgb * match_w[..., None]

    return np.clip(out, 0, 255).astype(np.uint8)
