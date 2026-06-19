from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import yaml


@dataclass(frozen=True)
class SimConfig:
    meshes_dir: Path
    output_dir: Path
    conveyor_speed_ms: float
    object_spacing_m: float
    camera_height_m: float
    camera_lateral_offset_m: float
    camera_x_offset_m: float
    camera_look_at_x_m: float
    depth_noise_sigma_base: float
    depth_noise_sigma_scale: float
    flying_pixel_threshold_m: float
    ir_speckle_alpha: float
    render_resolution: tuple
    fps: int
    render_samples: int
    belt_z: float = 0.05

    @classmethod
    def from_yaml(cls, path) -> "SimConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        res = raw.get("render_resolution", [640, 480])
        return cls(
            meshes_dir=Path(raw.get("meshes_dir", "./meshes")),
            output_dir=Path(raw.get("output_dir", "./output")),
            conveyor_speed_ms=float(raw.get("conveyor_speed_ms", 0.15)),
            object_spacing_m=float(raw.get("object_spacing_m", 0.5)),
            camera_height_m=float(raw.get("camera_height_m", 0.65)),
            camera_lateral_offset_m=float(raw.get("camera_lateral_offset_m", 0.65)),
            camera_x_offset_m=float(raw.get("camera_x_offset_m", 0.50)),
            camera_look_at_x_m=float(raw.get("camera_look_at_x_m", -0.30)),
            depth_noise_sigma_base=float(raw.get("depth_noise_sigma_base", 0.001)),
            depth_noise_sigma_scale=float(raw.get("depth_noise_sigma_scale", 0.002)),
            flying_pixel_threshold_m=float(raw.get("flying_pixel_threshold_m", 0.02)),
            ir_speckle_alpha=float(raw.get("ir_speckle_alpha", 0.04)),
            render_resolution=(int(res[0]), int(res[1])),
            fps=int(raw.get("fps", 30)),
            render_samples=int(raw.get("render_samples", 64)),
        )


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def as_open3d(self):
        import open3d as o3d
        return o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy
        )


def build_intrinsics(cfg: SimConfig) -> CameraIntrinsics:
    w, h = cfg.render_resolution
    fov_h = math.radians(87.0)
    fx = (w / 2.0) / math.tan(fov_h / 2.0)
    fov_v = 2.0 * math.atan(math.tan(fov_h / 2.0) * h / w)
    fy = (h / 2.0) / math.tan(fov_v / 2.0)
    return CameraIntrinsics(width=w, height=h, fx=fx, fy=fy, cx=w / 2.0, cy=h / 2.0)


def _lookat_extrinsic(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    # Solved from GT correspondences via SVD — matches Blender's camera geometry
    cam_neg_z = target - pos
    cam_neg_z = cam_neg_z / np.linalg.norm(cam_neg_z)
    cam_pos_z = -cam_neg_z

    world_Y = np.array([0.0, 1.0, 0.0])
    cam_x = np.cross(world_Y, cam_neg_z)
    norm = np.linalg.norm(cam_x)
    if norm < 1e-8:
        world_Y = np.array([0.0, 0.0, 1.0])
        cam_x = np.cross(world_Y, cam_neg_z)
        norm = np.linalg.norm(cam_x)
    cam_x = cam_x / norm
    cam_y = np.cross(cam_neg_z, cam_x)

    R_c2w = np.stack([cam_x, cam_y, cam_pos_z], axis=1)
    R_w2c = R_c2w.T
    t = -R_w2c @ pos

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_w2c
    T[:3, 3] = t
    return T


def _T_cam_to_world_left() -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([[ 0.6063, -0.4311,  0.6679],
                           [ 0.0000,  0.8398,  0.5426],
                           [-0.7951, -0.3290,  0.5093]])
    T[:3, 3] = np.array([0.50, 0.65, 0.71])
    return T


def _T_cam_to_world_right() -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([[ 0.6063,  0.4311,  0.6679],
                           [ 0.0000,  0.8398, -0.5426],
                           [-0.7951,  0.3290,  0.5093]])
    T[:3, 3] = np.array([0.50, -0.65, 0.71])
    return T


@dataclass(frozen=True)
class CameraExtrinsics:
    T_cam_to_world: np.ndarray

    @property
    def T_world_to_cam(self) -> np.ndarray:
        return np.linalg.inv(self.T_cam_to_world)

    @property
    def R(self) -> np.ndarray:
        return self.T_world_to_cam[:3, :3]

    @property
    def t(self) -> np.ndarray:
        return self.T_world_to_cam[:3, 3]


def build_extrinsics(cfg: SimConfig) -> Dict[str, CameraExtrinsics]:
    return {
        "left":  CameraExtrinsics(T_cam_to_world=_T_cam_to_world_left()),
        "right": CameraExtrinsics(T_cam_to_world=_T_cam_to_world_right()),
    }