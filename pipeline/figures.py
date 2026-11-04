#!/usr/bin/env python3
"""Assemble browser captures into the requested review sheets."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
HEAD = (150, 0, 874, 720)
KEYS = ['up-left', 'up', 'up-right', 'left', 'front', 'right', 'down-left', 'down', 'down-right']
KOREAN = {
    'front': '정면', 'left': '왼쪽', 'right': '오른쪽', 'up': '위', 'down': '아래',
    'up-left': '왼쪽 위', 'up-right': '오른쪽 위', 'down-left': '왼쪽 아래', 'down-right': '오른쪽 아래',
}
FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
]


def font(size):
    for name in FONT_CANDIDATES:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def open_rgb(path):
    image = Image.open(path).convert('RGBA')
    background = Image.new('RGBA', image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert('RGB')


def video_in_art(frame_number, transform):
    path = ROOT / 'source/references/seedance_frames640' / f'frame_{frame_number:04d}.png'
    frame = np.asarray(Image.open(path).convert('RGB'))
    matrix = np.asarray(transform, dtype=np.float32)
    return Image.fromarray(cv2.warpAffine(frame, matrix, (1024, 1536), flags=cv2.INTER_LINEAR,
                                           borderValue=(255, 255, 255)))


def draw_key_lines(image, key, targets, offset=(0, 0), color=(30, 110, 255), width=3):
    points = targets[key]
    draw = ImageDraw.Draw(image)
    ox, oy = offset

    def point(name):
        x, y = points[name]
        return x - ox, y - oy

    def line(names, close=False):
        found = [point(name) for name in names if name in points]
        if close and found:
            found.append(found[0])
        if len(found) > 1:
            draw.line(found, fill=color, width=width)

    line(['ct_l_brow', 'ct_l_eye', 'ct_l_nose', 'ct_l_mouth', 'ct_l_jaw', 'chin',
          'ct_r_jaw', 'ct_r_mouth', 'ct_r_nose', 'ct_r_eye', 'ct_r_brow'])
    for side in 'lr':
        first = f'eye_{side}_out' if side == 'l' else f'eye_{side}_in'
        last = f'eye_{side}_in' if side == 'l' else f'eye_{side}_out'
        line([first, f'eye_{side}_up1', f'eye_{side}_up2', f'eye_{side}_up3', last,
              f'eye_{side}_lo3', f'eye_{side}_lo2', f'eye_{side}_lo1'], close=True)
        line([f'brow_{side}_a', f'brow_{side}_arch', f'brow_{side}_b'])
    line(['mouth_l', 'mouth_c', 'mouth_r'])
    line(['hl_l', 'hl_lm', 'hl_c', 'hl_rm', 'hl_r'])
    line([f'sk{i:02d}' for i in range(13)])


def crop_head(image):
    return image.crop(HEAD)


def make_grid(columns, rows, titles, cell, output, labels=None, gap=24, title_height=44):
    width, height = cell
    result = Image.new('RGB', (columns * width + (columns - 1) * gap,
                               rows * (height + title_height) + (rows - 1) * gap), 'white')
    draw = ImageDraw.Draw(result)
    heading_font = font(28)
    row_font = font(26)
    for column, title in enumerate(titles):
        draw.text((column * (width + gap) + 8, 7), title, fill='black', font=heading_font)
    return result, row_font


def paste_tile(sheet, image, column, row, cell, gap=24, title_height=44, label=None, row_font=None):
    width, height = cell
    x = column * (width + gap)
    y = title_height + row * (height + title_height + gap)
    if label:
        ImageDraw.Draw(sheet).text((x + 8, y + 3), label, fill='black', font=row_font or font(26))
    sheet.paste(image.resize(cell, Image.Resampling.LANCZOS), (x, y + title_height - 6))


def save_front_diff(state_dir, output_dir, still_path):
    still = open_rgb(still_path)
    rendered = open_rgb(state_dir / 'front_hair.png')
    diff = np.abs(np.asarray(still).astype(np.int16) - np.asarray(rendered).astype(np.int16))
    diff = np.clip(diff * 8, 0, 255).astype(np.uint8)
    cell = (1024, 1536)
    sheet, row_font = make_grid(3, 1, ['원화', '새 화면', '차이 × 8'], cell, output_dir / 'front_diff.png', title_height=44)
    paste_tile(sheet, still, 0, 0, cell, label=None, row_font=row_font)
    paste_tile(sheet, rendered, 1, 0, cell, label=None, row_font=row_font)
    paste_tile(sheet, Image.fromarray(diff), 2, 0, cell, label=None, row_font=row_font)
    sheet.save(output_dir / 'front_diff.png')


def save_nine(state_dir, output_dir, data):
    cell = (724, 720)
    sheet, row_font = make_grid(3, 9, ['영상', '새 화면', '겹침'], cell, output_dir / 'nine_nohair.png')
    for row, key in enumerate(KEYS):
        frame = video_in_art(data['frames'][key], data['S'])
        viewer = open_rgb(state_dir / f'dir_{key}.png')
        viewer_head = crop_head(viewer)
        overlay = viewer_head.copy()
        draw_key_lines(overlay, key, data['targets'], offset=HEAD[:2])
        paste_tile(sheet, crop_head(frame), 0, row, cell, label=KOREAN[key], row_font=row_font)
        paste_tile(sheet, viewer_head, 1, row, cell, row_font=row_font)
        paste_tile(sheet, crop_head(overlay), 2, row, cell, row_font=row_font)
    sheet.save(output_dir / 'nine_nohair.png')


def eye_crop(targets, key):
    points = targets[key]
    values = [value for name, value in points.items() if name.startswith(('eye_l_', 'eye_r_'))]
    xs, ys = zip(*values)
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    return int(cx - 280), int(cy - 150), int(cx + 280), int(cy + 150)


def blink_state(state_dir, key, open_value):
    if open_value == 1:
        return open_rgb(state_dir / f'dir_{key}.png')
    name = f'{open_value:.2f}'.replace('.', '_')
    return open_rgb(state_dir / f'blink_{key}_{name}.png')


def save_blink(state_dir, output_dir, data):
    opens = [1, .7, .4, .23, 0]
    cell = (560, 300)
    sheet, row_font = make_grid(5, 9, [f'눈 뜬 정도 {value}' for value in opens], cell, output_dir / 'blink.png')
    transition_values = [.27, .25, .23, .21]
    transition, transition_font = make_grid(4, 9, [f'{value:.2f}' for value in transition_values], cell,
                                           output_dir / 'blink_transition.png')
    for row, key in enumerate(KEYS):
        box = eye_crop(data['targets'], key)
        for column, value in enumerate(opens):
            paste_tile(sheet, blink_state(state_dir, key, value).crop(box), column, row, cell,
                       label=KOREAN[key] if column == 0 else None, row_font=row_font)
        for column, value in enumerate(transition_values):
            paste_tile(transition, blink_state(state_dir, key, value).crop(box), column, row, cell,
                       label=KOREAN[key] if column == 0 else None, row_font=transition_font)
    sheet.save(output_dir / 'blink.png')
    transition.save(output_dir / 'blink_transition.png')


def save_bands(state_dir, output_dir):
    steps = [0, .2, .4, .6, .8, 1]
    tracks = {'left': (-1, 0), 'up': (0, 1), 'upleft': (-1, 1)}
    for name, (du, dv) in tracks.items():
        cell = (560, 560)
        sheet, row_font = make_grid(6, 2, [f'{value:.1f}' for value in steps], cell,
                                    output_dir / f'band_{name}.png')
        previous = None
        for column, value in enumerate(steps):
            if value == 0:
                image = open_rgb(state_dir / 'front_nohair.png')
            elif value == 1:
                endpoint = 'left' if du and not dv else 'up' if dv and not du else 'up-left'
                image = open_rgb(state_dir / f'dir_{endpoint}.png')
            else:
                step_name = f'{value:.1f}'.replace('.', '_')
                image = open_rgb(state_dir / f'band_{name}_{step_name}.png')
            tile = image.crop((232, 120, 792, 680))
            paste_tile(sheet, tile, column, 0, cell,
                       label=('정면에서 ' + {'left': '왼쪽 끝', 'up': '위 끝', 'upleft': '왼쪽 위 끝'}[name]) if column == 0 else None,
                       row_font=row_font)
            if previous is not None:
                current = np.asarray(image).astype(np.int16)
                difference = np.clip(np.abs(current - previous).max(-1) * 3, 0, 255).astype(np.uint8)
                paste_tile(sheet, Image.fromarray(difference).crop((232, 120, 792, 680)), column, 1, cell,
                           label='앞 단계와 차이' if column == 1 else None, row_font=row_font)
            previous = np.asarray(image).astype(np.int16)
        sheet.save(output_dir / f'band_{name}.png')


def build_figures(state_dir, output_dir, still_path):
    state_dir = Path(state_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / 'source/keyshapes/points_art_keyshapes.json').read_text())
    save_front_diff(state_dir, output_dir, still_path)
    save_nine(state_dir, output_dir, data)
    save_blink(state_dir, output_dir, data)
    save_bands(state_dir, output_dir)
