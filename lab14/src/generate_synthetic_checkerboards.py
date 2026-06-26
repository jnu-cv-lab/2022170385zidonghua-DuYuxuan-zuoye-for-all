#!/usr/bin/env python3
"""Generate synthetic checkerboard calibration images with OpenCV projection."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "synthetic_raw"
RESULTS_DIR = ROOT / "output" / "results"

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 960
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
BOARD_SQUARES = (10, 7)
INNER_CORNERS = (9, 6)
SQUARE_SIZE_MM = 25.0

K_TRUE = np.array(
    [
        [900.0, 0.0, 640.0],
        [0.0, 920.0, 480.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
D_TRUE = np.array([0.08, -0.03, 0.001, 0.0005, 0.01], dtype=np.float64)


POSES = [
    ("center_front", 0, 0, 0, 0, 0, 620),
    ("left", 4, -8, 2, -150, 10, 650),
    ("right", -3, 9, -2, 150, -10, 650),
    ("upper", -10, 2, 1, 0, -115, 660),
    ("lower", 10, -2, -1, 0, 120, 660),
    ("upper_left", -8, -10, 4, -145, -105, 690),
    ("upper_right", -8, 10, -4, 145, -105, 690),
    ("lower_left", 8, -10, -4, -145, 105, 690),
    ("lower_right", 8, 10, 4, 145, 105, 690),
    ("near_center", 2, -3, 6, 0, 5, 500),
    ("far_center", -2, 3, -6, 0, -5, 850),
    ("yaw_left", 2, -22, 0, -40, 0, 680),
    ("yaw_right", -2, 22, 0, 40, 0, 680),
    ("pitch_down", 20, 0, 0, 0, 20, 690),
    ("pitch_up", -20, 0, 0, 0, -20, 690),
    ("roll_clockwise", 0, 0, 16, 15, 0, 640),
    ("roll_counterclockwise", 0, 0, -16, -15, 0, 640),
    ("near_left_tilt", 12, -16, 9, -95, 45, 560),
    ("near_right_tilt", -12, 16, -9, 95, -45, 560),
    ("far_upper_roll", -14, -8, 12, -45, -90, 810),
    ("far_lower_roll", 14, 8, -12, 45, 90, 810),
    ("wide_upper_right", -6, 18, 10, 165, -130, 780),
    ("wide_lower_left", 6, -18, -10, -165, 130, 780),
    ("diagonal_close", 16, -12, 14, -70, -55, 540),
]


def euler_degrees_to_rvec(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Convert x/y/z Euler angles in degrees to an OpenCV rotation vector."""
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    rotation_matrix = rot_z @ rot_y @ rot_x
    rvec, _ = cv2.Rodrigues(rotation_matrix)
    return rvec.reshape(3, 1)


def make_grid_points() -> np.ndarray:
    """Return all checkerboard square grid intersections centered on the board."""
    cols, rows = BOARD_SQUARES
    board_width = cols * SQUARE_SIZE_MM
    board_height = rows * SQUARE_SIZE_MM
    points = []
    for y in range(rows + 1):
        for x in range(cols + 1):
            points.append(
                [
                    x * SQUARE_SIZE_MM - board_width / 2.0,
                    y * SQUARE_SIZE_MM - board_height / 2.0,
                    0.0,
                ]
            )
    return np.array(points, dtype=np.float32)


def project_points(points_3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, k: np.ndarray) -> np.ndarray:
    projected, _ = cv2.projectPoints(points_3d, rvec, tvec, k, D_TRUE)
    return projected.reshape(-1, 2)


def validate_pose(projected_grid: np.ndarray, margin: float = 18.0) -> None:
    min_xy = projected_grid.min(axis=0)
    max_xy = projected_grid.max(axis=0)
    if (
        min_xy[0] < margin
        or min_xy[1] < margin
        or max_xy[0] > IMAGE_WIDTH - margin
        or max_xy[1] > IMAGE_HEIGHT - margin
    ):
        raise ValueError(
            "Projected checkerboard is not fully visible: "
            f"min={min_xy.tolist()}, max={max_xy.tolist()}"
        )


def draw_checkerboard(rvec: np.ndarray, tvec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw a distorted perspective checkerboard by projecting square corners."""
    scale = 2
    canvas_width = IMAGE_WIDTH * scale
    canvas_height = IMAGE_HEIGHT * scale
    k_scaled = K_TRUE.copy()
    k_scaled[0, :] *= scale
    k_scaled[1, :] *= scale

    image = np.full((canvas_height, canvas_width, 3), 205, dtype=np.uint8)
    cols, rows = BOARD_SQUARES
    grid_points = make_grid_points()
    projected = project_points(grid_points, rvec, tvec, k_scaled)

    def idx(x: int, y: int) -> int:
        return y * (cols + 1) + x

    for y in range(rows):
        for x in range(cols):
            polygon = np.array(
                [
                    projected[idx(x, y)],
                    projected[idx(x + 1, y)],
                    projected[idx(x + 1, y + 1)],
                    projected[idx(x, y + 1)],
                ],
                dtype=np.int32,
            )
            color_value = 248 if (x + y) % 2 == 0 else 18
            cv2.fillConvexPoly(image, polygon, (color_value, color_value, color_value), lineType=cv2.LINE_AA)

    outer_polygon = np.array(
        [
            projected[idx(0, 0)],
            projected[idx(cols, 0)],
            projected[idx(cols, rows)],
            projected[idx(0, rows)],
        ],
        dtype=np.int32,
    )
    cv2.polylines(image, [outer_polygon], True, (8, 8, 8), thickness=2 * scale, lineType=cv2.LINE_AA)

    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    noise = rng.normal(0.0, 0.7, image.shape).astype(np.float32)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(14)
    grid_points = make_grid_points()
    pose_records = []

    for index, (name, rx, ry, rz, tx, ty, tz) in enumerate(POSES, start=1):
        rvec = euler_degrees_to_rvec(rx, ry, rz)
        tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)
        projected_for_validation = project_points(grid_points, rvec, tvec, K_TRUE)
        validate_pose(projected_for_validation)
        image = draw_checkerboard(rvec, tvec, rng)

        filename = f"calibration_{index:02d}.png"
        cv2.imwrite(str(RAW_DIR / filename), image)
        pose_records.append(
            {
                "image": filename,
                "pose_name": name,
                "euler_degrees_xyz": [rx, ry, rz],
                "rvec": rvec.reshape(-1).tolist(),
                "tvec_mm": tvec.reshape(-1).tolist(),
            }
        )

    parameters = {
        "camera_model": "synthetic pinhole camera with radial and tangential distortion",
        "K_true": K_TRUE.tolist(),
        "D_true_k1_k2_p1_p2_k3": D_TRUE.tolist(),
        "image_size": {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT},
        "square_size_mm": SQUARE_SIZE_MM,
        "board_squares": {"columns": BOARD_SQUARES[0], "rows": BOARD_SQUARES[1]},
        "inner_corners": {"columns": INNER_CORNERS[0], "rows": INNER_CORNERS[1]},
        "image_count": len(POSES),
        "poses": pose_records,
    }
    with (RESULTS_DIR / "true_camera_parameters.json").open("w", encoding="utf-8") as file:
        json.dump(parameters, file, indent=2)

    print(f"Generated {len(POSES)} synthetic checkerboard images in {RAW_DIR}")
    print(f"Saved true camera parameters to {RESULTS_DIR / 'true_camera_parameters.json'}")


if __name__ == "__main__":
    main()
