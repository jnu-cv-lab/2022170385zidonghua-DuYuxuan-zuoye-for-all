"""Implement high-dimensional Rotary Position Embedding with NumPy."""

import os
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = TASK_DIR / "output"
MPL_CONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np


SEQ_LEN = 8
D_MODEL = 8
RANDOM_SEED = 42


def get_rope_frequencies(d_model: int) -> np.ndarray:
    """Return RoPE frequencies for each 2D dimension group."""
    if d_model % 2 != 0:
        raise ValueError("d_model must be even.")

    group_indices = np.arange(d_model // 2)
    return 1.0 / np.power(10000.0, (2 * group_indices) / d_model)


def apply_rope_to_vector(x: np.ndarray, pos: int) -> np.ndarray:
    """Apply RoPE rotation to one high-dimensional vector at position pos."""
    vector = np.asarray(x, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("x must be a one-dimensional vector.")
    if vector.shape[0] % 2 != 0:
        raise ValueError("x dimension must be even.")

    d_model = vector.shape[0]
    frequencies = get_rope_frequencies(d_model)
    angles = pos * frequencies
    cos_values = np.cos(angles)
    sin_values = np.sin(angles)

    rotated = np.empty_like(vector)
    x_even = vector[0::2]
    x_odd = vector[1::2]
    rotated[0::2] = x_even * cos_values - x_odd * sin_values
    rotated[1::2] = x_even * sin_values + x_odd * cos_values
    return rotated


def apply_rope_to_sequence(X: np.ndarray) -> np.ndarray:
    """Apply RoPE to each vector in a sequence matrix by its position."""
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("X must be a two-dimensional matrix.")
    if matrix.shape[1] % 2 != 0:
        raise ValueError("d_model must be even.")

    rotated = np.empty_like(matrix)
    for pos in range(matrix.shape[0]):
        rotated[pos] = apply_rope_to_vector(matrix[pos], pos)
    return rotated


def save_heatmap(matrix: np.ndarray, output_path: Path, title: str) -> None:
    """Save a heatmap for a matrix."""
    plt.figure(figsize=(7, 4))
    plt.imshow(matrix, aspect="auto", cmap="coolwarm")
    plt.colorbar(label="Value")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_norm_comparison(
    positions: np.ndarray,
    norms_before: np.ndarray,
    norms_after: np.ndarray,
    output_path: Path,
) -> None:
    """Save a line chart comparing vector norms before and after RoPE."""
    plt.figure(figsize=(7, 4))
    plt.plot(positions, norms_before, marker="o", label="Before RoPE")
    plt.plot(positions, norms_after, marker="s", linestyle="--", label="After RoPE")
    plt.xlabel("Position")
    plt.ylabel("Vector norm")
    plt.title("Vector Norm Comparison Before and After RoPE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(RANDOM_SEED)
    X = np.random.randn(SEQ_LEN, D_MODEL)
    X_rope = apply_rope_to_sequence(X)

    norms_before = np.linalg.norm(X, axis=1)
    norms_after = np.linalg.norm(X_rope, axis=1)
    norm_differences = np.abs(norms_before - norms_after)
    positions = np.arange(SEQ_LEN)

    print(f"Original matrix X shape: {X.shape}")
    print(f"RoPE matrix X_rope shape: {X_rope.shape}")
    print("Position 0 before RoPE:")
    print(np.array2string(X[0], precision=6, suppress_small=True))
    print("Position 0 after RoPE:")
    print(np.array2string(X_rope[0], precision=6, suppress_small=True))
    print("Position 1 before RoPE:")
    print(np.array2string(X[1], precision=6, suppress_small=True))
    print("Position 1 after RoPE:")
    print(np.array2string(X_rope[1], precision=6, suppress_small=True))
    print(f"Position 0 unchanged check: {np.allclose(X[0], X_rope[0], atol=1e-10)}")
    print("Norm difference by position:")
    for pos, diff in zip(positions, norm_differences):
        print(f"position {pos}: {diff:.12f}")
    print(f"All vector norms preserved: {np.allclose(norms_before, norms_after, atol=1e-10)}")

    np.savetxt(OUTPUT_DIR / "input_matrix.csv", X, delimiter=",", fmt="%.10f")
    np.savetxt(OUTPUT_DIR / "rope_output_matrix.csv", X_rope, delimiter=",", fmt="%.10f")

    norm_check = np.column_stack((positions, norms_before, norms_after, norm_differences))
    np.savetxt(
        OUTPUT_DIR / "rope_norm_check.csv",
        norm_check,
        delimiter=",",
        header="position,norm_before,norm_after,norm_difference",
        comments="",
        fmt=["%d", "%.10f", "%.10f", "%.12f"],
    )

    save_heatmap(X, OUTPUT_DIR / "rope_input_heatmap.png", "Input Matrix Heatmap Before RoPE")
    save_heatmap(X_rope, OUTPUT_DIR / "rope_output_heatmap.png", "Output Matrix Heatmap After RoPE")
    save_norm_comparison(
        positions,
        norms_before,
        norms_after,
        OUTPUT_DIR / "rope_norm_comparison.png",
    )

    print(f"Saved input matrix: {OUTPUT_DIR / 'input_matrix.csv'}")
    print(f"Saved RoPE output matrix: {OUTPUT_DIR / 'rope_output_matrix.csv'}")
    print(f"Saved norm check: {OUTPUT_DIR / 'rope_norm_check.csv'}")
    print(f"Saved input heatmap: {OUTPUT_DIR / 'rope_input_heatmap.png'}")
    print(f"Saved output heatmap: {OUTPUT_DIR / 'rope_output_heatmap.png'}")
    print(f"Saved norm comparison: {OUTPUT_DIR / 'rope_norm_comparison.png'}")


if __name__ == "__main__":
    main()
