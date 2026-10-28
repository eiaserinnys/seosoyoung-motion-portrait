#!/usr/bin/env python3
"""Build the registration-only ICT-FaceKit mesh bundle and visual check sheet."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

from fetch_ict import ICT_COMMIT_SHA


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent.parent
MODEL_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = WORKSPACE_ROOT / ".local/artifacts/ict-facekit" / ICT_COMMIT_SHA
OUTPUT_NPZ = ARTIFACT_DIR / "ict_head.npz"
NEUTRAL_NAME = "FaceXModel/generic_neutral_mesh.obj"
VERTEX_LIMIT = 11248
POLYGON_LIMIT = 11144


def read_obj(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    vertices: list[tuple[float, float, float]] = []
    polygons: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            if line.startswith("v "):
                fields = line.split()
                vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
            elif line.startswith("f "):
                face = []
                for item in line.split()[1:]:
                    index = int(item.split("/", 1)[0])
                    face.append(index - 1 if index > 0 else len(vertices) + index)
                polygons.append(face)
    return np.asarray(vertices, dtype=np.float32), polygons


def triangulate(polygons: list[list[int]], polygon_limit: int) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    for polygon in polygons[:polygon_limit]:
        if len(polygon) < 3:
            raise ValueError("ICT mesh contains a polygon with fewer than three vertices")
        for index in range(1, len(polygon) - 1):
            triangles.append((polygon[0], polygon[index], polygon[index + 1]))
    return np.asarray(triangles, dtype=np.int32)


def read_obj_vertices(path: Path, minimum_count: int) -> np.ndarray:
    vertices: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            if line.startswith("v "):
                fields = line.split()
                vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
                if len(vertices) == minimum_count:
                    break
    if len(vertices) != minimum_count:
        raise ValueError(f"{path.name} has fewer than {minimum_count} vertices")
    return np.asarray(vertices, dtype=np.float32)


def read_readme_regions(readme: str) -> list[dict[str, object]]:
    section = readme.split("### Face Area Details", 1)[0]
    rows: list[dict[str, object]] = []
    for line in section.splitlines():
        if not line.startswith("|") or "Geometry name" in line or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 6 or not cells[0].startswith("#") and cells[0] != "n/a":
            continue
        vertex_range = re.fullmatch(r"\[(\d+):(\d+)\]", cells[2])
        polygon_range = re.fullmatch(r"\[(\d+):(\d+)\]", cells[3])
        if not vertex_range or not polygon_range:
            continue
        rows.append(
            {
                "ordinal": cells[0],
                "geometry_name": cells[1],
                "vertex_indices_inclusive": [int(vertex_range[1]), int(vertex_range[2])],
                "polygon_indices_inclusive": [int(polygon_range[1]), int(polygon_range[2])],
                "vertex_count": int(cells[4]),
                "face_count": int(cells[5]),
            }
        )
    if len(rows) != 18:
        raise ValueError(f"README topology table yielded {len(rows)} rows; expected 18")
    return rows


def read_landmarks(readme: str) -> list[int]:
    match = re.search(
        r"Multi-PIE 68 point facial landmarks indices:\s*\n\s*([0-9,\s]+)", readme
    )
    if not match:
        raise ValueError("README does not contain the Multi-PIE 68 landmark list")
    indices = [int(value) for value in re.findall(r"\d+", match.group(1))]
    if len(indices) != 68:
        raise ValueError(f"README landmark list contains {len(indices)} ids; expected 68")
    return indices


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def camera_basis(yaw_degrees: float, pitch_degrees: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw, pitch = np.radians([yaw_degrees, pitch_degrees])
    view = np.array(
        [np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)],
        dtype=np.float32,
    )
    right = np.array([np.cos(yaw), 0.0, -np.sin(yaw)], dtype=np.float32)
    up = np.cross(view, right)
    return right, up, view


def draw_mesh(
    canvas: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    yaw: float,
    pitch: float,
    center: np.ndarray,
    scale: float,
    light: np.ndarray,
) -> None:
    height, width = canvas.shape[:2]
    right, up, view = camera_basis(yaw, pitch)
    relative = vertices - center
    screen_x = relative @ right
    screen_y = relative @ up
    depth = relative @ view
    normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    lengths = np.linalg.norm(normals, axis=1)
    nonzero = lengths > 1e-8
    normals[nonzero] /= lengths[nonzero, None]
    illumination = np.maximum(normals @ light, 0.0)
    shades = np.clip(72 + illumination * 150, 0, 255).astype(np.uint8)
    face_depth = depth[faces].mean(axis=1)

    for face_id in np.argsort(face_depth):
        points = faces[face_id]
        polygon = np.column_stack(
            (
                width / 2 + screen_x[points] * scale,
                height / 2 - screen_y[points] * scale,
            )
        ).round().astype(np.int32)
        value = int(shades[face_id])
        cv2.fillConvexPoly(canvas, polygon, (value, value, value), lineType=cv2.LINE_AA)


def render_check_sheet(
    neutral: np.ndarray,
    faces: np.ndarray,
    modes: np.ndarray,
    target: Path,
) -> None:
    cell = 360
    margin = 24
    center = (neutral.min(axis=0) + neutral.max(axis=0)) / 2
    extent = neutral.max(axis=0) - neutral.min(axis=0)
    scale = (cell - margin * 2) / max(float(extent[0]), float(extent[1]), float(extent[2]))
    light = np.array([-0.32, 0.52, 0.79], dtype=np.float32)
    light /= np.linalg.norm(light)
    views = [
        ("Front", 0.0, 0.0, neutral),
        ("Yaw 45", 45.0, 0.0, neutral),
        ("Side 90", 90.0, 0.0, neutral),
        ("Top", 0.0, 82.0, neutral),
        ("Back", 180.0, 0.0, neutral),
        ("Yaw 45 Pitch -30", 45.0, -30.0, neutral),
        ("Mode 0 -3", 0.0, 0.0, neutral - 3.0 * modes[0]),
        ("Mode 0 +3", 0.0, 0.0, neutral + 3.0 * modes[0]),
        ("Mode 1 -3", 0.0, 0.0, neutral - 3.0 * modes[1]),
        ("Mode 1 +3", 0.0, 0.0, neutral + 3.0 * modes[1]),
    ]
    sheet = np.full((cell * 2, cell * 5, 3), 248, dtype=np.uint8)
    for index, (label, yaw, pitch, mesh) in enumerate(views):
        row, column = divmod(index, 5)
        x, y = column * cell, row * cell
        panel = sheet[y : y + cell, x : x + cell]
        draw_mesh(panel, mesh, faces, yaw, pitch, center, scale, light)
        cv2.putText(panel, label, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (24, 24, 24), 1, cv2.LINE_AA)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), sheet):
        raise RuntimeError(f"Could not write {target}")


def main() -> None:
    manifest_path = MODEL_DIR / "ict_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("commit") != ICT_COMMIT_SHA:
        raise ValueError("ICT manifest commit does not match the pinned source revision")
    readme_path = ARTIFACT_DIR / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    regions = read_readme_regions(readme)
    landmarks = read_landmarks(readme)

    neutral_all, polygons = read_obj(ARTIFACT_DIR / NEUTRAL_NAME)
    if len(neutral_all) < VERTEX_LIMIT or len(polygons) < POLYGON_LIMIT:
        raise ValueError("ICT neutral mesh is smaller than the declared Face + Head and Neck region")
    neutral = neutral_all[:VERTEX_LIMIT]
    faces = triangulate(polygons, POLYGON_LIMIT)
    if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= VERTEX_LIMIT):
        raise ValueError("A selected surface triangle references a vertex outside the head region")

    identity_modes = np.empty((100, VERTEX_LIMIT, 3), dtype=np.float32)
    for mode_index in range(100):
        path = ARTIFACT_DIR / f"FaceXModel/identity{mode_index:03d}.obj"
        identity_vertices = read_obj_vertices(path, VERTEX_LIMIT)
        # ICT's official loader computes each mode as morph-target points minus
        # generic-neutral points, then adds weighted modes to the neutral mesh.
        identity_modes[mode_index] = identity_vertices - neutral

    write_json(
        MODEL_DIR / "ict_regions.json",
        {
            "source_commit": ICT_COMMIT_SHA,
            "index_base": 0,
            "readme_topology_table": regions,
            "registration_surface": {
                "name": "Face + Head and Neck outer surface",
                "vertex_indices_inclusive": [0, 11247],
                "polygon_indices_inclusive": [0, 11143],
                "included_geometries": ["Face", "Head and Neck"],
                "excluded_internal_geometries": [
                    "Mouth socket",
                    "Eye socket left",
                    "Eye socket right",
                    "Gums and tongue",
                    "Teeth",
                    "Eyeball left",
                    "Eyeball right",
                    "Lacrimal fluid left",
                    "Lacrimal fluid right",
                    "Eye blend left",
                    "Eye blend right",
                    "Eye occlusion left",
                    "Eye occlusion right",
                    "Eyelashes left",
                    "Eyelashes right",
                ],
                "triangulation": "fan from the first polygon vertex; original polygon order retained",
            },
        },
    )
    write_json(
        MODEL_DIR / "ict_landmarks.json",
        {
            "source_commit": ICT_COMMIT_SHA,
            "index_base": 0,
            "convention": "Multi-PIE 68 point facial landmarks",
            "indices": landmarks,
        },
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        neutral_vertices=neutral,
        triangles=faces,
        identity_modes=identity_modes,
    )
    render_check_sheet(neutral, faces, identity_modes, MODEL_DIR / "sheets/ict-check.png")
    print(f"Vertices: {len(neutral)}; triangles: {len(faces)}; identity modes: {len(identity_modes)}")
    print(f"Saved compressed mesh: {OUTPUT_NPZ}")
    print(f"Saved region and landmark definitions under {MODEL_DIR}")
    print(f"Saved render check: {MODEL_DIR / 'sheets/ict-check.png'}")


if __name__ == "__main__":
    main()
