from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def render_before_after_comparison(
    existing_map: dict[str, np.ndarray],
    rebuilt_map: dict[str, np.ndarray],
    debug_dir: Path,
) -> None:
    def checker(w: int, h: int, cell: int = 12) -> Image.Image:
        y, x = np.indices((h, w))
        c = ((x // cell) + (y // cell)) % 2
        bg = np.where(c[..., None] == 0, 42, 58).astype(np.uint8)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = bg
        rgba[..., 3] = 255
        return Image.fromarray(rgba, "RGBA")

    compare_specs = [
        ("face-skin", "face-skin", (260, 200, 740, 635)),
        ("neck", "neck", (330, 520, 670, 820)),
        ("ear-left", "ear-left", (275, 395, 395, 535)),
        ("eye-left-sclera", "eye-left-sclera", (330, 370, 465, 485)),
        ("eye-left-iris", "eye-left-iris", (375, 395, 455, 475)),
        ("hair-top", "hair-top", (210, 0, 790, 420)),
        ("tail-left-hair", "tail-left-hair", (145, 485, 395, 815)),
        ("tail-left-tie", "tail-left-tie", (235, 495, 380, 595)),
    ]

    cell_w, cell_h = 260, 220
    grid = Image.new("RGBA", (4 * cell_w + 30, 4 * cell_h + 40), (18, 20, 26, 255))
    draw = ImageDraw.Draw(grid)

    for idx, (title, pid, box) in enumerate(compare_specs):
        row = idx // 2
        col_pair = (idx % 2) * 2
        for sub_idx, (tag, src_map) in enumerate([("BEFORE", existing_map), ("AFTER (SAM3+Inpaint)", rebuilt_map)]):
            arr = src_map[pid]
            im_crop = Image.fromarray(arr, "RGBA").crop(box)
            im_crop.thumbnail((cell_w - 16, cell_h - 36), Image.Resampling.LANCZOS)
            bg = checker(cell_w - 12, cell_h - 32, 10)
            ox = (bg.width - im_crop.width) // 2
            oy = (bg.height - im_crop.height) // 2
            bg.alpha_composite(im_crop, (ox, oy))

            gx = (col_pair + sub_idx) * cell_w + 15
            gy = row * cell_h + 30
            grid.paste(bg, (gx, gy))
            color = (240, 150, 150, 255) if sub_idx == 0 else (140, 235, 190, 255)
            draw.text((gx + 4, gy - 18), f"{tag}: {pid}", fill=color)

    grid.save(debug_dir / "before_after_parts_comparison.png")
