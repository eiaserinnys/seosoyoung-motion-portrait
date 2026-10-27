#!/usr/bin/env python3
"""Fetch the pinned ICT-FaceKit neutral mesh and identity shape modes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


ICT_REPOSITORY = "ICT-VGL/ICT-FaceKit"
ICT_COMMIT_SHA = "da5f95a607f5e6b37755b38d3385d7f2853732e5"
RAW_ROOT = f"https://raw.githubusercontent.com/{ICT_REPOSITORY}/{ICT_COMMIT_SHA}"
FILES = [
    "FaceXModel/generic_neutral_mesh.obj",
    *(f"FaceXModel/identity{i:03d}.obj" for i in range(100)),
    "README.md",
    "LICENSE",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent.parent
ARTIFACT_DIR = WORKSPACE_ROOT / ".local/artifacts/ict-facekit" / ICT_COMMIT_SHA
MANIFEST_PATH = Path(__file__).resolve().parent / "ict_manifest.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_file(relative_path: str) -> bytes:
    url = f"{RAW_ROOT}/{relative_path}"
    request = Request(url, headers={"User-Agent": "seosoyoung-motion-portrait-headmodel-inputs"})
    with urlopen(request, timeout=90) as response:
        if response.status != 200:
            raise RuntimeError(f"ICT-FaceKit download failed: {relative_path} ({response.status})")
        return response.read()


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for relative_path in FILES:
        content = fetch_file(relative_path)
        target = ARTIFACT_DIR / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        temporary.write_bytes(content)
        os.replace(temporary, target)
        records.append(
            {
                "path": relative_path,
                "sha256": sha256_bytes(content),
                "bytes": len(content),
            }
        )

    manifest = {
        "repository": ICT_REPOSITORY,
        "commit": ICT_COMMIT_SHA,
        "source": f"https://github.com/{ICT_REPOSITORY}/tree/{ICT_COMMIT_SHA}",
        "artifact_directory": str(ARTIFACT_DIR.relative_to(WORKSPACE_ROOT)),
        "files": records,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Fetched {len(records)} files for ICT-FaceKit {ICT_COMMIT_SHA}.")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Artifacts: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
