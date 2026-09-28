"""Lightweight reader for the released PEAR directory structure."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PEARFrame:
    """Paths and metadata associated with one annotated frame."""

    scene: str
    frame_id: int
    category: str
    instance: str
    occluded: bool
    rgb_path: Path
    depth_path: Path
    mask_path: Path
    pose_path: Path
    camera_extrinsic_path: Path
    intrinsics_path: Path
    mesh_path: Path
    object_to_world_path: Path


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


class PEARDataset:
    """Index PEAR through its canonical ``frames.csv`` file."""

    def __init__(
        self,
        root: str | Path,
        *,
        category: str | None = None,
        scene: str | None = None,
        occluded: bool | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        frame_index = self.root / "frames.csv"
        if not frame_index.is_file():
            raise FileNotFoundError(f"Missing PEAR frame index: {frame_index}")

        self._scene_metadata = self._read_scene_metadata()
        with frame_index.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        if category is not None:
            wanted = category.lower().replace("_", "")
            rows = [
                row
                for row in rows
                if self._scene_metadata[row["scene"]]["category"]
                .lower()
                .replace("_", "")
                == wanted
            ]
        if scene is not None:
            rows = [row for row in rows if row["scene"] == scene]
        if occluded is not None:
            rows = [row for row in rows if _as_bool(row["occluded"]) == occluded]
        self._rows = rows

    def _read_scene_metadata(self) -> dict[str, dict[str, Any]]:
        manifest_path = self.root / "dataset_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing PEAR manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata: dict[str, dict[str, Any]] = {}
        for entry in manifest["scenes"]:
            scene_manifest_path = self.root / entry["manifest"]
            scene_manifest = json.loads(
                scene_manifest_path.read_text(encoding="utf-8")
            )
            metadata[entry["scene"]] = scene_manifest
        return metadata

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[PEARFrame]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> PEARFrame:
        row = self._rows[index]
        scene = row["scene"]
        scene_root = self.root / "scenes" / scene
        metadata = self._scene_metadata[scene]
        files = metadata["files"]
        return PEARFrame(
            scene=scene,
            frame_id=int(row["frame_id"]),
            category=metadata["category"],
            instance=metadata["instance"],
            occluded=_as_bool(row["occluded"]),
            rgb_path=self.root / "scenes" / row["rgb"],
            depth_path=self.root / "scenes" / row["depth"],
            mask_path=self.root / "scenes" / row["mask"],
            pose_path=self.root / "scenes" / row["pose_m2c"],
            camera_extrinsic_path=(
                self.root / "scenes" / row["camera_extrinsic_c2w"]
            ),
            intrinsics_path=scene_root / files["intrinsics"],
            mesh_path=scene_root / files["mesh"],
            object_to_world_path=scene_root / files["object_to_world"],
        )

    def load(self, index: int) -> dict[str, Any]:
        """Decode one frame and return arrays alongside its metadata."""

        frame = self[index]
        return {
            "scene": frame.scene,
            "frame_id": frame.frame_id,
            "category": frame.category,
            "instance": frame.instance,
            "occluded": frame.occluded,
            "rgb": np.asarray(Image.open(frame.rgb_path).convert("RGB")),
            "depth_mm": np.asarray(Image.open(frame.depth_path)),
            "mask": np.asarray(Image.open(frame.mask_path)) > 0,
            "intrinsics": np.loadtxt(frame.intrinsics_path).reshape(3, 3),
            "object_to_camera": np.loadtxt(frame.pose_path).reshape(4, 4),
            "camera_to_world": np.loadtxt(
                frame.camera_extrinsic_path
            ).reshape(4, 4),
            "object_to_world": np.loadtxt(
                frame.object_to_world_path
            ).reshape(4, 4),
            "mesh_path": frame.mesh_path,
        }
