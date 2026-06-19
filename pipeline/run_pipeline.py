from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
from tqdm import tqdm

from pipeline.config import SimConfig, build_extrinsics, build_intrinsics
from pipeline.precompute_gallery import GalleryEntry
from pipeline.utils.camera import belt_segmentation_mask, depth_to_pointcloud
from pipeline.utils.registration import (
    RegistrationParams,
    icp_refine,
    preprocess,
    ransac_registration,
)



def load_depth(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(str(path)).astype(np.float32)


def load_frame_metadata(data_dir: Path) -> list[dict]:
    meta_path = data_dir / "metadata" / "objects.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path) as f:
        return json.load(f)



def build_observation_cloud(
    data_dir: Path,
    frame_id: int,
    intrinsics,
    extrinsics: dict,
    cfg: SimConfig,
    min_points: int = 200,
) -> o3d.geometry.PointCloud | None:
    clouds = []

    for side in ("left", "right"):
        depth_path = (
            data_dir / f"cam_{side}" / "depth" / f"{frame_id:06d}.npy"
        )
        depth = load_depth(depth_path)
        if depth is None:
            continue

        ext = extrinsics[side]
        mask = belt_segmentation_mask(
            depth,
            belt_z_world=cfg.belt_z + 0.01,
            extrinsics=ext,
            intrinsics=intrinsics,
        )

        if mask.sum() < min_points:
            continue

        pcd = depth_to_pointcloud(depth, intrinsics, ext, mask=mask)
        if len(pcd.points) >= min_points:
            clouds.append(pcd)

    if not clouds:
        return None

    merged = clouds[0]
    for c in clouds[1:]:
        merged += c

    # Light downsampling to remove duplicate points from overlapping FOVs
    merged = merged.voxel_down_sample(0.003)

    # belt segmentation mask can't distinguish frame structure from object; tight box removes false positives
    BELT_HALF_WIDTH_M = 0.30
    z_lo = cfg.belt_z + 0.005  # just above belt surface
    z_hi = cfg.belt_z + 0.28   # tallest object + margin
    pts = np.asarray(merged.points)
    keep = (
        (pts[:, 2] >= z_lo) &
        (pts[:, 2] <= z_hi) &
        (np.abs(pts[:, 1]) <= BELT_HALF_WIDTH_M)
    )
    merged = merged.select_by_index(np.where(keep)[0])

    # belt frame at z≈0.08–0.15m overwhelms the object; two RANSAC plane passes strip it
    for _ in range(2):
        if len(merged.points) <= 500:
            break
        _, inliers = merged.segment_plane(
            distance_threshold=0.04,
            ransac_n=3,
            num_iterations=500,
        )
        remaining = merged.select_by_index(inliers, invert=True)
        if len(remaining.points) >= min_points:
            merged = remaining
        else:
            break

    # residual belt structure spans metres in X; real objects are compact
    pts = np.asarray(merged.points)
    if pts[:, 0].max() - pts[:, 0].min() > 0.80:
        return None

    if len(merged.points) < min_points:
        return None

    return merged


def classify_and_estimate_pose(
    observed: o3d.geometry.PointCloud,
    gallery: dict[str, GalleryEntry],
    params: RegistrationParams,
) -> tuple[str, np.ndarray, float] | None:
    # classify by RANSAC fitness (shape-discriminative); ICP fitness favours convex geometry so unreliable for classification
    observed_down, observed_fpfh = preprocess(observed, params)

    best_ransac_id: str | None = None
    best_ransac_fitness: float = -1.0
    best_ransac_transform: np.ndarray | None = None

    for obj_id, entry in gallery.items():
        ransac_result = ransac_registration(
            source_down=observed_down,
            target_down=entry.to_pcd_down(),
            source_fpfh=observed_fpfh,
            target_fpfh=entry.to_fpfh(),
            params=params,
        )

        if ransac_result.fitness > best_ransac_fitness:
            best_ransac_fitness = ransac_result.fitness
            best_ransac_id = obj_id
            best_ransac_transform = ransac_result.transformation

    if best_ransac_id is None or best_ransac_fitness < 0.05:
        return None

    icp_result = icp_refine(
        source=observed,
        target=gallery[best_ransac_id].to_pcd(),
        init_transform=best_ransac_transform,
        params=params,
    )

    if icp_result.fitness < 0.05:
        return None

    return best_ransac_id, icp_result.transformation, icp_result.fitness


def T_obs_to_world(T_obs_to_CAD: np.ndarray) -> np.ndarray:
    # obs cloud is already in world frame, so T_obs_to_CAD == T_world_to_CAD; invert for object pose
    return np.linalg.inv(T_obs_to_CAD)



def run(
    data_dir: Path,
    config_path: Path,
    gallery_path: Path,
    out_path: Path,
) -> None:
    cfg = SimConfig.from_yaml(config_path)
    intrinsics = build_intrinsics(cfg)
    extrinsics = build_extrinsics(cfg)

    with open(gallery_path, "rb") as f:
        gallery: dict[str, GalleryEntry] = pickle.load(f)

    params = RegistrationParams()
    frames_meta = load_frame_metadata(data_dir.parent)

    predictions = []

    for frame in tqdm(frames_meta, desc="Frames"):
        frame_id = frame["frame_id"]
        objects_in_scene = frame.get("objects_in_scene", [])

        if not objects_in_scene:
            predictions.append({"frame_id": frame_id, "prediction": None})
            continue

        observed = build_observation_cloud(
            data_dir, frame_id, intrinsics, extrinsics, cfg
        )

        if observed is None or len(observed.points) < 200:
            predictions.append({"frame_id": frame_id, "prediction": None})
            continue

        result = classify_and_estimate_pose(observed, gallery, params)

        if result is None:
            predictions.append({"frame_id": frame_id, "prediction": None})
            continue

        pred_id, T_obs_to_CAD, fitness = result
        T_world = T_obs_to_world(T_obs_to_CAD)

        predictions.append({
            "frame_id": frame_id,
            "prediction": {
                "object_id": pred_id,
                "world_transform": T_world.tolist(),
                "icp_fitness": fitness,
            },
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nSaved {len(predictions)} predictions → {out_path}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classification + pose estimation pipeline.")
    parser.add_argument("--data", type=Path, required=True, help="Path to output/ directory")
    parser.add_argument("--config", type=Path, default=Path("sim_config.yaml"))
    parser.add_argument("--gallery", type=Path, default=Path("gallery.pkl"))
    parser.add_argument("--out", type=Path, default=Path("predictions.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        data_dir=args.data,
        config_path=args.config,
        gallery_path=args.gallery,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()