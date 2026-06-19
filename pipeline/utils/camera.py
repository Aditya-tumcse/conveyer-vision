from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d

from pipeline.config import CameraExtrinsics, CameraIntrinsics


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    mask: np.ndarray | None = None,
    depth_min: float = 0.1,
    depth_max: float = 3.0,
) -> o3d.geometry.PointCloud:
    h, w = depth.shape
    assert (h, w) == (intrinsics.height, intrinsics.width)

    valid = np.isfinite(depth) & (depth >= depth_min) & (depth <= depth_max)
    if mask is not None:
        valid &= mask.astype(bool)

    us, vs = np.meshgrid(np.arange(w), np.arange(h))

    d = depth[valid]
    x_c = (us[valid] - intrinsics.cx) * d / intrinsics.fx
    y_c = -(vs[valid] - intrinsics.cy) * d / intrinsics.fy
    z_c = -d

    pts_cam = np.stack([x_c, y_c, z_c, np.ones_like(z_c)], axis=1)
    T_cam_to_world = extrinsics.T_cam_to_world
    pts_world = (T_cam_to_world @ pts_cam.T).T[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)
    return pcd


def belt_segmentation_mask(
    depth: np.ndarray,
    belt_z_world: float,
    extrinsics: CameraExtrinsics,
    intrinsics: CameraIntrinsics,
    max_object_height_m: float = 0.25,
    erode_px: int = 3,
) -> np.ndarray:
    h, w = depth.shape

    us, vs = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))

    dx = (us - intrinsics.cx) / intrinsics.fx
    dy = -(vs - intrinsics.cy) / intrinsics.fy

    R_cw = extrinsics.T_cam_to_world[:3, :3]
    t_cw = extrinsics.T_cam_to_world[:3, 3]

    dirs_cam = np.stack([dx, dy, -np.ones_like(dx)], axis=-1)
    dirs_world_z = (R_cw[2:3, :] @ dirs_cam.reshape(-1, 3).T).reshape(h, w)

    denom = dirs_world_z
    safe = denom < -0.25
    belt_depth = np.where(safe, (belt_z_world - t_cw[2]) / denom, np.nan)

    noise_margin = 0.01
    depth_upper = belt_depth - noise_margin
    depth_lower = depth_upper - max_object_height_m

    mask = (
        np.isfinite(depth) &
        np.isfinite(belt_depth) &
        (depth >= depth_lower) &
        (depth <= depth_upper)
    )

    if erode_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)
        )
        mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)

    return mask


def bbox_mask(
    depth: np.ndarray,
    bbox_xyxy: list,
    padding: int = 5,
) -> np.ndarray:
    h, w = depth.shape
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask