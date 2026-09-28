# PEAR

**Pose and dEformation of Agricultural pRoduce**

PEAR is an RGB-D benchmark designed to study the instance-specific geometric
deformation of agricultural produce. Its object-pose annotations also support
conventional 6D pose estimation and evaluation.

PEAR contains 9,607 annotated frames from 117 scenes of 39 produce instances
across eight categories: apple, avocado, banana, carrot, lemon, long pepper,
pear, and pumpkin. It provides RGB images, corrected depth, object masks,
camera intrinsics and extrinsics, 6D object poses, per-instance scanned meshes,
reconstructed point clouds, and occlusion annotations.

## Resources

- Dataset: https://huggingface.co/datasets/enci2/PEAR
- Paper: https://arxiv.org/abs/2603.27429
- Project page: https://kvantonikolas.github.io/PEAR-SEED/

## Download PEAR

PEAR is hosted independently of the SEED implementation. After installing the
Hugging Face client, download the complete dataset with:

```bash
python -m pip install -U huggingface_hub
hf download enci2/PEAR --repo-type dataset --local-dir PEAR
```

## Minimal Setup

Create a small environment and install the loader:

```bash
conda create -n pear python=3.11 -y
conda activate pear

git clone https://github.com/kvantonikolas/PEAR-SEED.git
python -m pip install -e PEAR-SEED/PEAR
```

The loader requires only NumPy and Pillow; both are installed automatically.

## Example

Run the included example on the downloaded dataset:

```bash
python PEAR-SEED/PEAR/examples/load_frame.py \
  /path/to/PEAR \
  --category banana \
  --index 0
```

Or use the loader directly:

```python
from pear_dataset import PEARDataset

dataset = PEARDataset("/path/to/PEAR", category="banana")
sample = dataset.load(0)

print(sample["rgb"].shape)
print(sample["depth_mm"].shape)
print(sample["object_to_camera"])
print(sample["mesh_path"])
```

`dataset[index]` returns a lightweight record with paths and metadata.
`dataset.load(index)` additionally decodes the images and transformation
matrices.

## Dataset Layout

```text
PEAR/
|-- README.md
|-- LICENSE
|-- dataset_manifest.json
|-- frames.csv
|-- scenes/
|   `-- scene_XXX_Y/
|       |-- scene_manifest.json
|       |-- cam_K.txt
|       |-- object_pose_m2w.txt
|       |-- rgb/
|       |-- depth/
|       |-- masks/
|       |-- poses/
|       |-- camera_extrinsics/
|       `-- mesh/
|-- pointclouds/
`-- occl_index/
```

The scene suffix denotes the capture condition: `_0` is isolated, `_1` is
cluttered, and `_2` contains occlusion. Use `frames.csv` and each scene's
`valid_frame_ids` instead of assuming that all numerical frame IDs exist.

The camera convention is OpenCV (`+x` right, `+y` down, `+z` forward), and the
provided transformations satisfy:

```text
T_object_to_camera = inverse(T_camera_to_world) @ T_object_to_world
```

Transform translations and point clouds are in metres. Depth PNG values and
mesh vertices are in millimetres; multiply mesh vertices by `0.001` before
applying the provided transformations.

## License

The loader and example code are licensed under the
[Apache License 2.0](LICENSE). The PEAR dataset is distributed separately under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and should be
attributed using the citation below.

## Citation

```bibtex
@inproceedings{Chatzis2026PEAR,
  title={Mind the Shape Gap: A Benchmark and Baseline for Deformation-Aware 6D Pose Estimation of Agricultural Produce},
  author={Nikolas Chatzis and Angeliki Tsinouka and Katerina Papadimitriou and Niki Efthymiou and Marios Glytsos and George Retsinas and Paris Oikonomou and Gerasimos Potamianos and Petros Maragos and Panagiotis Paraskevas Filntisis},
  booktitle={Proceedings of the International Conference on Intelligent Robots and Systems (IROS)},
  year={2026}
}
```
