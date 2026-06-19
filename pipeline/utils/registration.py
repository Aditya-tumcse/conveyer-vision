from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d



@dataclass
class RegistrationParams:
    # Preprocessing
    voxel_size: float = 0.005

    # Normal estimation
    normal_radius_factor: float = 2.0
    normal_max_nn: int = 30

    # ISS keypoints
    iss_salient_radius_factor: float = 6.0
    iss_non_max_radius_factor: float = 4.0
    iss_min_neighbors: int = 5
    iss_min_keypoints: int = 50

    # FPFH
    fpfh_radius_factor: float = 5.0
    fpfh_max_nn: int = 100

    # RANSAC global registration
    ransac_distance_factor: float = 1.5
    ransac_max_iterations: int = 100000
    ransac_confidence: float = 0.999
    ransac_n_correspondence: int = 3

    # ICP refinement
    icp_distance_factor: float = 1.0
    icp_max_iterations: int = 100
    icp_relative_fitness: float = 1e-6
    icp_relative_rmse: float = 1e-6



def _estimate_normals(pcd: o3d.geometry.PointCloud, params: RegistrationParams) -> None:
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=params.normal_radius_factor * params.voxel_size,
            max_nn=params.normal_max_nn,
        )
    )


def _compute_fpfh(
    pcd: o3d.geometry.PointCloud,
    params: RegistrationParams,
) -> o3d.pipelines.registration.Feature:
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=params.fpfh_radius_factor * params.voxel_size,
            max_nn=params.fpfh_max_nn,
        ),
    )


def preprocess(
    pcd: o3d.geometry.PointCloud,
    params: RegistrationParams,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    # falls back to voxel downsampling if ISS yields fewer than iss_min_keypoints (common on flat objects)
    voxel = params.voxel_size

    _estimate_normals(pcd, params)

    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd,
        salient_radius=params.iss_salient_radius_factor * voxel,
        non_max_radius=params.iss_non_max_radius_factor * voxel,
        min_neighbors=params.iss_min_neighbors,
    )

    if len(keypoints.points) >= params.iss_min_keypoints:
        _estimate_normals(keypoints, params)
        sparse = keypoints
    else:
        sparse = pcd.voxel_down_sample(voxel)
        _estimate_normals(sparse, params)

    fpfh = _compute_fpfh(sparse, params)
    return sparse, fpfh



@dataclass
class RegistrationResult:
    transformation: np.ndarray   # (4, 4) source → target
    fitness: float               # inlier ratio
    inlier_rmse: float


def ransac_registration(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
    params: RegistrationParams,
) -> RegistrationResult:
    # o3d aligns source onto target; passing (observed, CAD) gives T_obs_in_CAD_frame
    distance_threshold = params.ransac_distance_factor * params.voxel_size

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=params.ransac_n_correspondence,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            params.ransac_max_iterations,
            params.ransac_confidence,
        ),
    )

    return RegistrationResult(
        transformation=np.array(result.transformation),
        fitness=result.fitness,
        inlier_rmse=result.inlier_rmse,
    )



def icp_refine(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    params: RegistrationParams,
) -> RegistrationResult:
    distance_threshold = params.icp_distance_factor * params.voxel_size

    if not target.has_normals():
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=params.normal_radius_factor * params.voxel_size,
                max_nn=params.normal_max_nn,
            )
        )

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=params.icp_max_iterations,
            relative_fitness=params.icp_relative_fitness,
            relative_rmse=params.icp_relative_rmse,
        ),
    )

    return RegistrationResult(
        transformation=np.array(result.transformation),
        fitness=result.fitness,
        inlier_rmse=result.inlier_rmse,
    )
