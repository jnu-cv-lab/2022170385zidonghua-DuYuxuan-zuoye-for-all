"""Rotate two-dimensional vectors with NumPy."""

import os
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = TASK_DIR / "output"
MPL_CONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np


TEST_VECTOR = np.array([1.0, 0.0])
TEST_ANGLES_DEGREE = [0, 45, 90, 180, -90]


def rotation_matrix(theta: float) -> np.ndarray:
    """Return the 2D rotation matrix for angle theta in radians."""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array(
        [
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta],
        ],
        dtype=np.float64,
    )


def rotate_vector(v: np.ndarray, theta: float) -> np.ndarray:
    """Rotate a 2D vector by theta radians."""
    vector = np.asarray(v, dtype=np.float64)
    if vector.shape != (2,):
        raise ValueError("Input vector must have shape (2,).")
    return rotation_matrix(theta) @ vector


def plot_vectors(results: list[dict[str, float]], output_path: Path) -> None:
    """Plot the original vector and all rotated vectors."""
    plt.figure(figsize=(6, 6))
    origin = np.array([0.0, 0.0])

    plt.quiver(
        origin[0],
        origin[1],
        TEST_VECTOR[0],
        TEST_VECTOR[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="black",
        width=0.008,
        label="original [1, 0]",
    )

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for result, color in zip(results, colors):
        plt.quiver(
            origin[0],
            origin[1],
            result["rotated_x"],
            result["rotated_y"],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            alpha=0.85,
            label=f"{result['angle_degree']:.0f} deg",
        )

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(-1.3, 1.3)
    plt.ylim(-1.3, 1.3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D Vector Rotation at Multiple Angles")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_90_degree_rotation(rotated_vector: np.ndarray, output_path: Path) -> None:
    """Plot a dedicated diagram for the 90-degree rotation."""
    plt.figure(figsize=(6, 6))
    origin = np.array([0.0, 0.0])

    plt.quiver(
        origin[0],
        origin[1],
        TEST_VECTOR[0],
        TEST_VECTOR[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="black",
        width=0.008,
        label="original [1, 0]",
    )
    plt.quiver(
        origin[0],
        origin[1],
        rotated_vector[0],
        rotated_vector[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="tab:green",
        width=0.008,
        label="rotated 90 deg",
    )

    theta = np.linspace(0, np.pi / 2, 100)
    plt.plot(0.25 * np.cos(theta), 0.25 * np.sin(theta), color="tab:green", linewidth=1.5)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D Vector Rotation by 90 Degrees")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_results_csv(results: list[dict[str, float]], output_path: Path) -> None:
    """Save rotation results to a CSV file."""
    header = "angle_degree,input_x,input_y,rotated_x,rotated_y"
    rows = [
        [
            result["angle_degree"],
            result["input_x"],
            result["input_y"],
            result["rotated_x"],
            result["rotated_y"],
        ]
        for result in results
    ]
    np.savetxt(output_path, rows, delimiter=",", header=header, comments="", fmt="%.10f")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    rotated_90 = None

    for angle_degree in TEST_ANGLES_DEGREE:
        theta = np.deg2rad(angle_degree)
        matrix = rotation_matrix(theta)
        rotated = rotate_vector(TEST_VECTOR, theta)

        print(f"Angle: {angle_degree} degrees")
        print("Rotation matrix:")
        print(np.array2string(matrix, precision=6, suppress_small=True))
        print(f"Rotated vector: {np.array2string(rotated, precision=6, suppress_small=True)}")
        print()

        if angle_degree == 90:
            rotated_90 = rotated

        results.append(
            {
                "angle_degree": float(angle_degree),
                "input_x": float(TEST_VECTOR[0]),
                "input_y": float(TEST_VECTOR[1]),
                "rotated_x": float(rotated[0]),
                "rotated_y": float(rotated[1]),
            }
        )

    expected_results = {
        0: np.array([1.0, 0.0]),
        90: np.array([0.0, 1.0]),
        180: np.array([-1.0, 0.0]),
        -90: np.array([0.0, -1.0]),
    }
    print("Key result verification:")
    for angle_degree, expected in expected_results.items():
        actual = rotate_vector(TEST_VECTOR, np.deg2rad(angle_degree))
        is_correct = np.allclose(actual, expected, atol=1e-10)
        print(f"{angle_degree:>4} deg: {actual} close to {expected} -> {is_correct}")

    csv_path = OUTPUT_DIR / "vector_rotation_results.csv"
    multiple_angles_path = OUTPUT_DIR / "vector_rotation_multiple_angles.png"
    rotation_90_path = OUTPUT_DIR / "vector_rotation_90deg.png"

    save_results_csv(results, csv_path)
    plot_vectors(results, multiple_angles_path)
    if rotated_90 is None:
        raise RuntimeError("The 90-degree rotation result was not generated.")
    plot_90_degree_rotation(rotated_90, rotation_90_path)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved multiple-angle plot: {multiple_angles_path}")
    print(f"Saved 90-degree plot: {rotation_90_path}")


if __name__ == "__main__":
    main()
