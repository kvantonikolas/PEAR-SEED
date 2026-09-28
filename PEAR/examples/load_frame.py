"""Load one PEAR frame and print its principal annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

from pear_dataset import PEARDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--category")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    dataset = PEARDataset(args.dataset_root, category=args.category)
    sample = dataset.load(args.index)
    print(f"scene: {sample['scene']}")
    print(f"frame: {sample['frame_id']}")
    print(f"category: {sample['category']}")
    print(f"instance: {sample['instance']}")
    print(f"occluded: {sample['occluded']}")
    print(f"RGB shape: {sample['rgb'].shape}")
    print(f"depth shape: {sample['depth_mm'].shape}")
    print(f"mesh: {sample['mesh_path']}")
    print("object-to-camera pose:")
    print(sample["object_to_camera"])


if __name__ == "__main__":
    main()
