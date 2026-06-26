#!/usr/bin/env python3
"""Run camera calibration from synthetic checkerboard images."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "synthetic_raw"
OUTPUT_DIR = ROOT / "output"
CORNERS_DIR = OUTPUT_DIR / "corners"
UNDISTORTED_DIR = OUTPUT_DIR / "undistorted"
COMPARE_DIR = OUTPUT_DIR / "compare"
SAMPLES_DIR = OUTPUT_DIR / "samples"
RESULTS_DIR = OUTPUT_DIR / "results"

PATTERN_SIZE = (9, 6)
SQUARE_SIZE_MM = 25.0
MIN_VALID_IMAGES = 15


def ensure_directories() -> None:
    for directory in [CORNERS_DIR, UNDISTORTED_DIR, COMPARE_DIR, SAMPLES_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def make_object_points() -> np.ndarray:
    cols, rows = PATTERN_SIZE
    object_points = np.zeros((rows * cols, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_points *= SQUARE_SIZE_MM
    return object_points


def image_files() -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg"}
    return sorted(path for path in RAW_DIR.iterdir() if path.suffix.lower() in suffixes)


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    height = int(round(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_grid(images: list[np.ndarray], columns: int, cell_width: int) -> np.ndarray:
    resized = [resize_to_width(image, cell_width) for image in images]
    cell_height = max(image.shape[0] for image in resized)
    normalized = []
    for image in resized:
        if image.shape[0] < cell_height:
            pad = cell_height - image.shape[0]
            image = cv2.copyMakeBorder(image, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
        normalized.append(image)

    rows = []
    for start in range(0, len(normalized), columns):
        row_images = normalized[start : start + columns]
        while len(row_images) < columns:
            row_images.append(np.full_like(normalized[0], 245))
        rows.append(np.hstack(row_images))
    return np.vstack(rows)


def labeled_side_by_side(left: np.ndarray, right: np.ndarray, left_label: str, right_label: str) -> np.ndarray:
    height = max(left.shape[0], right.shape[0])
    if left.shape[0] != height:
        left = cv2.copyMakeBorder(left, 0, height - left.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    if right.shape[0] != height:
        right = cv2.copyMakeBorder(right, 0, height - right.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    compare = np.hstack([left, right])
    label_bar = np.full((52, compare.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(label_bar, left_label, (24, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(
        label_bar,
        right_label,
        (left.shape[1] + 24, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([label_bar, compare])


def save_results_text(
    total_count: int,
    successes: list[str],
    failures: list[str],
    rms: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    per_image_errors: list[dict[str, float | str]],
) -> None:
    mean_error = float(np.mean([row["mean_error_px"] for row in per_image_errors]))
    with (RESULTS_DIR / "calibration_results.txt").open("w", encoding="utf-8") as file:
        file.write("Camera Calibration Using a Checkerboard Pattern\n")
        file.write("================================================\n\n")
        file.write(f"Total images: {total_count}\n")
        file.write(f"Detection success: {len(successes)}\n")
        file.write(f"Detection failure: {len(failures)}\n")
        file.write(f"Failed images: {', '.join(failures) if failures else 'None'}\n\n")
        file.write(f"RMS reprojection error from cv2.calibrateCamera: {rms:.8f} px\n")
        file.write(f"Manual mean reprojection error: {mean_error:.8f} px\n\n")
        file.write("Camera matrix K:\n")
        file.write(np.array2string(camera_matrix, precision=8))
        file.write("\n\nDistortion coefficients D [k1, k2, p1, p2, k3]:\n")
        file.write(np.array2string(distortion.reshape(-1), precision=8))
        file.write("\n")


def main() -> None:
    ensure_directories()
    files = image_files()
    if not files:
        raise FileNotFoundError(f"No calibration images found in {RAW_DIR}")

    object_template = make_object_points()
    object_points_list: list[np.ndarray] = []
    image_points_list: list[np.ndarray] = []
    success_files: list[Path] = []
    failed_files: list[str] = []
    corner_images: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.001,
    )
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            failed_files.append(path.name)
            print(f"[FAIL] Could not read {path.name}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, flags)

        if not found:
            failed_files.append(path.name)
            print(f"[FAIL] Corners not found in {path.name}")
            continue

        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points_list.append(object_template.copy())
        image_points_list.append(refined)
        success_files.append(path)

        drawn = image.copy()
        cv2.drawChessboardCorners(drawn, PATTERN_SIZE, refined, found)
        cv2.imwrite(str(CORNERS_DIR / f"{path.stem}_corners.png"), drawn)
        corner_images.append(drawn)
        print(f"[OK] {path.name}")

    if len(object_points_list) < MIN_VALID_IMAGES:
        summary_path = RESULTS_DIR / "detection_failures.txt"
        with summary_path.open("w", encoding="utf-8") as file:
            file.write(f"Valid detections: {len(object_points_list)}\n")
            file.write(f"Required minimum: {MIN_VALID_IMAGES}\n")
            file.write("Failed images:\n")
            for name in failed_files:
                file.write(f"{name}\n")
        raise RuntimeError(
            f"Only {len(object_points_list)} valid images were detected. "
            "Please regenerate or adjust the synthetic images."
        )

    assert image_size is not None
    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points_list,
        image_points_list,
        image_size,
        None,
        None,
    )
    distortion = distortion.reshape(-1)[:5]

    per_image_errors: list[dict[str, float | str]] = []
    for path, obj_points, img_points, rvec, tvec in zip(
        success_files,
        object_points_list,
        image_points_list,
        rvecs,
        tvecs,
    ):
        projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, distortion)
        projected = projected.reshape(-1, 2)
        detected = img_points.reshape(-1, 2)
        distances = np.linalg.norm(projected - detected, axis=1)
        per_image_errors.append(
            {
                "image": path.name,
                "mean_error_px": float(np.mean(distances)),
                "max_error_px": float(np.max(distances)),
                "rms_error_px": float(np.sqrt(np.mean(distances**2))),
            }
        )

    with (RESULTS_DIR / "per_image_reprojection_errors.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "mean_error_px", "max_error_px", "rms_error_px"])
        writer.writeheader()
        writer.writerows(per_image_errors)

    np.savetxt(RESULTS_DIR / "camera_matrix_K.csv", camera_matrix, delimiter=",", fmt="%.10f")
    np.savetxt(RESULTS_DIR / "distortion_coefficients_D.csv", distortion.reshape(1, -1), delimiter=",", fmt="%.10f")

    mean_manual_error = float(np.mean([row["mean_error_px"] for row in per_image_errors]))
    results_json = {
        "pattern_size_inner_corners": {"columns": PATTERN_SIZE[0], "rows": PATTERN_SIZE[1]},
        "square_size_mm": SQUARE_SIZE_MM,
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "total_images": len(files),
        "detection_success_count": len(success_files),
        "detection_failure_count": len(failed_files),
        "failed_images": failed_files,
        "rms_reprojection_error_px": float(rms),
        "manual_mean_reprojection_error_px": mean_manual_error,
        "camera_matrix_K": camera_matrix.tolist(),
        "distortion_coefficients_D_k1_k2_p1_p2_k3": distortion.tolist(),
        "rvecs": [rvec.reshape(-1).tolist() for rvec in rvecs],
        "tvecs_mm": [tvec.reshape(-1).tolist() for tvec in tvecs],
        "per_image_errors": per_image_errors,
    }
    with (RESULTS_DIR / "calibration_results.json").open("w", encoding="utf-8") as file:
        json.dump(results_json, file, indent=2)
    save_results_text(len(files), [path.name for path in success_files], failed_files, rms, camera_matrix, distortion, per_image_errors)

    first_image = cv2.imread(str(success_files[0]))
    undistorted = cv2.undistort(first_image, camera_matrix, distortion)
    undistorted_path = UNDISTORTED_DIR / f"{success_files[0].stem}_undistorted.png"
    cv2.imwrite(str(undistorted_path), undistorted)

    compare = labeled_side_by_side(first_image, undistorted, "Original distorted image", "Undistorted image")
    compare_path = COMPARE_DIR / "undistort_compare.png"
    cv2.imwrite(str(compare_path), compare)

    raw_sample_indices = np.linspace(0, len(success_files) - 1, 4, dtype=int)
    raw_images = [cv2.imread(str(success_files[index])) for index in raw_sample_indices]
    cv2.imwrite(str(SAMPLES_DIR / "sample_4_raw_images.png"), make_grid(raw_images, columns=2, cell_width=520))
    cv2.imwrite(str(SAMPLES_DIR / "sample_2_corner_images.png"), make_grid(corner_images[:2], columns=2, cell_width=520))
    shutil.copyfile(compare_path, SAMPLES_DIR / "sample_undistort_compare.png")

    print("\nCamera calibration summary")
    print("==========================")
    print(f"Total images: {len(files)}")
    print(f"Detection success: {len(success_files)}")
    print(f"Detection failure: {len(failed_files)}")
    print(f"Failed images: {', '.join(failed_files) if failed_files else 'None'}")
    print("Camera matrix K:")
    print(camera_matrix)
    print("Distortion coefficients D [k1, k2, p1, p2, k3]:")
    print(distortion)
    print(f"RMS reprojection error: {rms:.8f} px")
    print(f"Manual mean reprojection error: {mean_manual_error:.8f} px")
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
