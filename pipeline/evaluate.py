from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh

from pipeline.precompute_gallery import BOUNDING_CUBE_M, load_and_normalise_mesh



def compute_add(pts: np.ndarray, T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])   # (N, 4)
    pts_pred = (T_pred @ pts_h.T).T[:, :3]
    pts_gt   = (T_gt   @ pts_h.T).T[:, :3]
    return float(np.mean(np.linalg.norm(pts_pred - pts_gt, axis=1)))



def evaluate(
    predictions_path: Path,
    gt_path: Path,
    meshes_dir: Path | None,
    add_threshold_ratio: float = 0.1,
) -> None:
    with open(predictions_path) as f:
        predictions: list[dict] = json.load(f)
    with open(gt_path) as f:
        gt_frames: list[dict] = json.load(f)

    gt_by_frame: dict[int, dict] = {}
    for frame in gt_frames:
        objs = frame.get("objects_in_scene", [])
        if objs:
            gt_by_frame[frame["frame_id"]] = objs[0]  # one object per frame

    model_pts: dict[str, np.ndarray] = {}
    if meshes_dir is not None:
        for fp in meshes_dir.iterdir():
            if fp.suffix.lower() in {".obj", ".stl"}:
                mesh, _ = load_and_normalise_mesh(fp)
                pts, _ = trimesh.sample.sample_surface(mesh, 2000)
                model_pts[fp.stem] = pts.astype(np.float64)

    total = 0
    correct_cls = 0
    cls_total: dict[str, int] = defaultdict(int)
    cls_correct: dict[str, int] = defaultdict(int)
    add_errors: list[float] = []
    add_correct = 0

    for pred_frame in predictions:
        frame_id = pred_frame["frame_id"]
        pred = pred_frame.get("prediction")

        if frame_id not in gt_by_frame:
            continue

        gt = gt_by_frame[frame_id]
        gt_id = gt["object_id"]
        gt_T = np.array(gt["world_transform"])

        total += 1
        cls_total[gt_id] += 1

        if pred is None:
            continue

        pred_id = pred["object_id"]
        pred_T = np.array(pred["world_transform"])

        if pred_id == gt_id:
            correct_cls += 1
            cls_correct[gt_id] += 1

            # ADD only makes sense for correct classification
            if pred_id in model_pts:
                diameter = BOUNDING_CUBE_M  # approx; all meshes fit in 0.22m cube
                threshold = add_threshold_ratio * diameter
                add = compute_add(model_pts[pred_id], pred_T, gt_T)
                add_errors.append(add)
                if add < threshold:
                    add_correct += 1

    print(f"\n{'='*55}")
    print(f"  Evaluation over {total} frames with GT objects")
    print(f"{'='*55}")
    print(f"\nClassification accuracy: {correct_cls}/{total} = {correct_cls/max(total,1):.1%}")

    print("\nPer-class accuracy:")
    for cls in sorted(cls_total):
        n = cls_total[cls]
        c = cls_correct[cls]
        print(f"  {cls:<20s} {c}/{n} = {c/max(n,1):.1%}")

    if add_errors:
        n_add = len(add_errors)
        print(f"\nADD (on correctly classified frames, n={n_add}):")
        print(f"  Mean:    {np.mean(add_errors)*1000:.1f} mm")
        print(f"  Median:  {np.median(add_errors)*1000:.1f} mm")
        print(f"  <10% dia ({0.1*BOUNDING_CUBE_M*1000:.0f}mm): {add_correct}/{n_add} = {add_correct/n_add:.1%}")
    else:
        print("\nADD: no model point files found or no correct classifications — skipped")

    print(f"\n{'='*55}\n")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pose estimation predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--meshes", type=Path, default=None,
                        help="Mesh directory for ADD metric (optional)")
    parser.add_argument("--add-threshold", type=float, default=0.1,
                        help="ADD threshold as fraction of object diameter (default 0.1)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.predictions, args.ground_truth, args.meshes, args.add_threshold)


if __name__ == "__main__":
    main()
