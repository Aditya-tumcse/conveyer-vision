from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from pipeline.utils.registration import RegistrationParams, preprocess



@dataclass
class GalleryEntry:  # numpy arrays only — o3d objects are not picklable
    object_id: str
    pts: np.ndarray           # (N, 3) full sampled points
    pts_down: np.ndarray      # (M, 3) keypoint positions
    normals_down: np.ndarray  # (M, 3) keypoint normals
    fpfh_data: np.ndarray     # (33, M) FPFH descriptor matrix
    scale: float              # Blender normalisation scale

    def to_pcd(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pts)
        return pcd

    def to_pcd_down(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pts_down)
        pcd.normals = o3d.utility.Vector3dVector(self.normals_down)
        return pcd

    def to_fpfh(self) -> o3d.pipelines.registration.Feature:
        feat = o3d.pipelines.registration.Feature()
        feat.data = self.fpfh_data
        return feat



BOUNDING_CUBE_M = 0.22   # matches run_sim.py `0.22 / max_dim`


def load_and_normalise_mesh(filepath: Path) -> tuple[trimesh.Trimesh, float]:
    mesh = trimesh.load(str(filepath), force="mesh")
    extents = mesh.bounding_box.extents
    max_dim = float(extents.max())
    if max_dim < 1e-9:
        raise ValueError(f"Degenerate mesh: {filepath}")
    scale = BOUNDING_CUBE_M / max_dim
    mesh.apply_scale(scale)

    z_min = mesh.vertices[:, 2].min()
    mesh.apply_translation([0.0, 0.0, -z_min])

    return mesh, scale


def sample_pointcloud(mesh: trimesh.Trimesh, n_points: int = 50_000) -> o3d.geometry.PointCloud:
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    return pcd



MESH_EXTENSIONS = {".obj", ".stl"}


def build_gallery(
    meshes_dir: Path,
    params: RegistrationParams,
    n_sample_points: int = 50_000,
) -> dict[str, GalleryEntry]:
    gallery: dict[str, GalleryEntry] = {}

    mesh_files = [
        p for p in sorted(meshes_dir.iterdir())
        if p.suffix.lower() in MESH_EXTENSIONS
    ]
    if not mesh_files:
        raise FileNotFoundError(f"No OBJ/STL files found in {meshes_dir}")

    for filepath in mesh_files:
        object_id = filepath.stem
        print(f"  Processing {object_id} ...")

        mesh, scale = load_and_normalise_mesh(filepath)
        pcd = sample_pointcloud(mesh, n_points=n_sample_points)
        pcd_down, fpfh = preprocess(pcd, params)

        gallery[object_id] = GalleryEntry(
            object_id=object_id,
            pts=np.asarray(pcd.points),
            pts_down=np.asarray(pcd_down.points),
            normals_down=np.asarray(pcd_down.normals),
            fpfh_data=np.asarray(fpfh.data),
            scale=scale,
        )
        print(f"    {len(pcd.points):,} pts → {len(pcd_down.points):,} keypoints")

    return gallery



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute FPFH gallery from CAD meshes.")
    parser.add_argument("--meshes", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("sim_config.yaml"))
    parser.add_argument("--out", type=Path, default=Path("gallery.pkl"))
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--n-sample", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = RegistrationParams(voxel_size=args.voxel_size)

    print(f"Building gallery from {args.meshes} ...")
    gallery = build_gallery(args.meshes, params, n_sample_points=args.n_sample)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(gallery, f)

    print(f"\nSaved gallery with {len(gallery)} entries → {args.out}")


if __name__ == "__main__":
    main()
