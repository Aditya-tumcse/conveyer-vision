# conveyer-vision-pipeline

6-DoF pose estimation for objects on a conveyor belt. Stereo depth cameras feed into FPFH + RANSAC classification followed by ICP pose refinement against a precomputed CAD gallery.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager
- Python 3.11+

## Folder structure

```
conveyer-vision-pipeline/
├── pipeline/
│   ├── config.py               # SimConfig, camera intrinsics/extrinsics
│   ├── precompute_gallery.py   # builds FPFH descriptor gallery from CAD meshes
│   ├── run_pipeline.py         # main inference: classify + estimate 6-DoF pose
│   ├── evaluate.py             # classification accuracy + ADD pose error
│   └── utils/
│       ├── camera.py           # depth unprojection, belt segmentation mask
│       └── registration.py     # RANSAC global registration + ICP refinement
├── input/
│   ├── meshes/                 # CAD meshes (.obj / .stl) — not in git
│   └── images/                 # depth frames + ground-truth metadata — not in git
│       ├── cam_left/depth/
│       ├── cam_right/depth/
│       └── metadata/objects.json
├── artifacts/                  # generated outputs — not in git
│   ├── gallery.pkl
│   └── predictions.json
├── sim_config.yaml
└── pyproject.toml
```

## Install

```bash
uv sync
```

This creates a virtual environment at `.venv/` and installs all dependencies. To include dev tools (pytest, ruff):

```bash
uv sync --group dev
```

## Workflow

### 1. Precompute the gallery

Processes each CAD mesh in `input/meshes/` into a descriptor gallery used at inference time.

```bash
uv run precompute-gallery \
  --meshes input/meshes/ \
  --config sim_config.yaml \
  --out artifacts/gallery.pkl
```

This only needs to re-run when the mesh set changes.

### 2. Run the pipeline

Classifies each frame and estimates a 6-DoF world-frame pose.

```bash
uv run run-pipeline \
  --data input/images/ \
  --config sim_config.yaml \
  --gallery artifacts/gallery.pkl \
  --out artifacts/predictions.json
```

### 3. Evaluate

Computes classification accuracy and ADD pose error against ground-truth metadata.

```bash
uv run evaluate-pipeline \
  --predictions artifacts/predictions.json \
  --ground-truth input/images/metadata/objects.json \
  --meshes input/meshes/
```

## Configuration

All tunable parameters live in `sim_config.yaml`. The most commonly adjusted:

| Key | Default | Description |
|-----|---------|-------------|
| `meshes_dir` | `input/meshes` | CAD mesh directory |
| `output_dir` | `input/images` | Simulation output directory |
| `render_samples` | `64` | Blender Cycles samples (speed vs. quality) |
| `conveyor_speed_ms` | `0.15` | Belt speed in m/s |

## Author

Developed by Aditya Sai Srinivas
