"""Compare E+pos and RoPE position injection with NumPy."""

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


def sinusoidal_position_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Generate sinusoidal position encoding with shape (seq_len, d_model)."""
    if d_model % 2 != 0:
        raise ValueError("d_model must be even.")

    positions = np.arange(seq_len)[:, np.newaxis]
    even_dimensions = np.arange(0, d_model, 2)
    div_terms = np.power(10000.0, even_dimensions / d_model)

    encoding = np.zeros((seq_len, d_model), dtype=np.float64)
    encoding[:, 0::2] = np.sin(positions / div_terms)
    encoding[:, 1::2] = np.cos(positions / div_terms)
    return encoding


def get_rope_frequencies(d_model: int) -> np.ndarray:
    """Return one RoPE frequency for each adjacent 2D dimension group."""
    if d_model % 2 != 0:
        raise ValueError("d_model must be even.")

    group_indices = np.arange(d_model // 2)
    return 1.0 / np.power(10000.0, (2 * group_indices) / d_model)


def apply_rope_to_vector(x: np.ndarray, pos: int) -> np.ndarray:
    """Apply RoPE to one vector at a specific position."""
    vector = np.asarray(x, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("x must be a one-dimensional vector.")
    if vector.shape[0] % 2 != 0:
        raise ValueError("x dimension must be even.")

    frequencies = get_rope_frequencies(vector.shape[0])
    angles = pos * frequencies
    cos_values = np.cos(angles)
    sin_values = np.sin(angles)

    even_values = vector[0::2]
    odd_values = vector[1::2]
    rotated = np.empty_like(vector)
    rotated[0::2] = even_values * cos_values - odd_values * sin_values
    rotated[1::2] = even_values * sin_values + odd_values * cos_values
    return rotated


def apply_rope_to_sequence(E: np.ndarray) -> np.ndarray:
    """Apply RoPE to every embedding vector by its sequence position."""
    matrix = np.asarray(E, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("E must be a two-dimensional matrix.")
    if matrix.shape[1] % 2 != 0:
        raise ValueError("d_model must be even.")

    rotated = np.empty_like(matrix)
    for pos in range(matrix.shape[0]):
        rotated[pos] = apply_rope_to_vector(matrix[pos], pos)
    return rotated


def save_heatmap(matrix: np.ndarray, output_path: Path, title: str) -> None:
    """Save a matrix heatmap."""
    plt.figure(figsize=(7, 4))
    plt.imshow(matrix, aspect="auto", cmap="coolwarm")
    plt.colorbar(label="Value")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_norm_comparison_plot(
    positions: np.ndarray,
    norm_E: np.ndarray,
    norm_E_plus_pos: np.ndarray,
    norm_RoPE: np.ndarray,
    output_path: Path,
) -> None:
    """Save a line plot comparing vector norms."""
    plt.figure(figsize=(7, 4))
    plt.plot(positions, norm_E, marker="o", label="E")
    plt.plot(positions, norm_E_plus_pos, marker="s", linestyle="--", label="E+pos")
    plt.plot(positions, norm_RoPE, marker="^", linestyle="-.", label="RoPE")
    plt.xlabel("Position")
    plt.ylabel("Vector norm")
    plt.title("Norm Comparison of E, E+pos, and RoPE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_position3_component_plot(
    E: np.ndarray,
    E_plus_pos: np.ndarray,
    E_rope: np.ndarray,
    output_path: Path,
) -> None:
    """Save a grouped bar chart for vector components at position 3."""
    position = 3
    dimensions = np.arange(E.shape[1])
    width = 0.25

    plt.figure(figsize=(8, 4.5))
    plt.bar(dimensions - width, E[position], width=width, label="E")
    plt.bar(dimensions, E_plus_pos[position], width=width, label="E+pos")
    plt.bar(dimensions + width, E_rope[position], width=width, label="RoPE")
    plt.xlabel("Dimension")
    plt.ylabel("Component value")
    plt.title("Position 3 Component Comparison")
    plt.xticks(dimensions)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_norm_csv(
    positions: np.ndarray,
    norm_E: np.ndarray,
    norm_E_plus_pos: np.ndarray,
    norm_RoPE: np.ndarray,
    output_path: Path,
) -> None:
    """Save vector norm comparison values to CSV."""
    diff_E_plus_pos = np.abs(norm_E_plus_pos - norm_E)
    diff_RoPE = np.abs(norm_RoPE - norm_E)
    data = np.column_stack(
        (
            positions,
            norm_E,
            norm_E_plus_pos,
            norm_RoPE,
            diff_E_plus_pos,
            diff_RoPE,
        )
    )
    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header="position,norm_E,norm_E_plus_pos,norm_RoPE,diff_E_plus_pos,diff_RoPE",
        comments="",
        fmt=["%d", "%.10f", "%.10f", "%.10f", "%.10f", "%.12f"],
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(RANDOM_SEED)
    E = np.random.randn(SEQ_LEN, D_MODEL)
    pos = sinusoidal_position_encoding(SEQ_LEN, D_MODEL)
    E_plus_pos = E + pos
    E_rope = apply_rope_to_sequence(E)

    norm_E = np.linalg.norm(E, axis=1)
    norm_E_plus_pos = np.linalg.norm(E_plus_pos, axis=1)
    norm_RoPE = np.linalg.norm(E_rope, axis=1)
    diff_E_plus_pos = np.abs(norm_E_plus_pos - norm_E)
    diff_RoPE = np.abs(norm_RoPE - norm_E)
    positions = np.arange(SEQ_LEN)

    print(f"E shape: {E.shape}")
    print(f"pos shape: {pos.shape}")
    print(f"E_plus_pos shape: {E_plus_pos.shape}")
    print(f"E_rope shape: {E_rope.shape}")

    for position in (0, 3):
        print(f"Position {position} E:")
        print(np.array2string(E[position], precision=6, suppress_small=True))
        print(f"Position {position} pos:")
        print(np.array2string(pos[position], precision=6, suppress_small=True))
        print(f"Position {position} E+pos:")
        print(np.array2string(E_plus_pos[position], precision=6, suppress_small=True))
        print(f"Position {position} RoPE:")
        print(np.array2string(E_rope[position], precision=6, suppress_small=True))

    e_plus_pos_changes_norm = np.any(diff_E_plus_pos > 1e-10)
    rope_preserves_norm = np.allclose(norm_RoPE, norm_E, atol=1e-10)
    rope_pos0_unchanged = np.allclose(E_rope[0], E[0], atol=1e-10)

    print("Verification conclusions:")
    print(f"E+pos changes vector length: {e_plus_pos_changes_norm}")
    print(f"RoPE preserves vector length: {rope_preserves_norm}")
    print(f"RoPE at pos=0 equals original vector: {rope_pos0_unchanged}")
    print("Norm comparison by position:")
    for position, diff_add, diff_rope in zip(positions, diff_E_plus_pos, diff_RoPE):
        print(
            f"position {position}: "
            f"diff_E_plus_pos={diff_add:.10f}, diff_RoPE={diff_rope:.12f}"
        )

    np.savetxt(OUTPUT_DIR / "embedding_matrix.csv", E, delimiter=",", fmt="%.10f")
    np.savetxt(OUTPUT_DIR / "position_encoding_matrix.csv", pos, delimiter=",", fmt="%.10f")
    np.savetxt(OUTPUT_DIR / "e_plus_pos_matrix.csv", E_plus_pos, delimiter=",", fmt="%.10f")
    np.savetxt(OUTPUT_DIR / "rope_matrix.csv", E_rope, delimiter=",", fmt="%.10f")
    save_norm_csv(positions, norm_E, norm_E_plus_pos, norm_RoPE, OUTPUT_DIR / "norm_comparison.csv")

    save_heatmap(E, OUTPUT_DIR / "embedding_heatmap.png", "Embedding Matrix Heatmap")
    save_heatmap(E_plus_pos, OUTPUT_DIR / "e_plus_pos_heatmap.png", "E+pos Matrix Heatmap")
    save_heatmap(E_rope, OUTPUT_DIR / "rope_heatmap.png", "RoPE Matrix Heatmap")
    save_norm_comparison_plot(
        positions,
        norm_E,
        norm_E_plus_pos,
        norm_RoPE,
        OUTPUT_DIR / "norm_comparison.png",
    )
    save_position3_component_plot(
        E,
        E_plus_pos,
        E_rope,
        OUTPUT_DIR / "position3_component_comparison.png",
    )

    print(f"Saved embedding matrix: {OUTPUT_DIR / 'embedding_matrix.csv'}")
    print(f"Saved position encoding matrix: {OUTPUT_DIR / 'position_encoding_matrix.csv'}")
    print(f"Saved E+pos matrix: {OUTPUT_DIR / 'e_plus_pos_matrix.csv'}")
    print(f"Saved RoPE matrix: {OUTPUT_DIR / 'rope_matrix.csv'}")
    print(f"Saved norm comparison: {OUTPUT_DIR / 'norm_comparison.csv'}")
    print(f"Saved embedding heatmap: {OUTPUT_DIR / 'embedding_heatmap.png'}")
    print(f"Saved E+pos heatmap: {OUTPUT_DIR / 'e_plus_pos_heatmap.png'}")
    print(f"Saved RoPE heatmap: {OUTPUT_DIR / 'rope_heatmap.png'}")
    print(f"Saved norm plot: {OUTPUT_DIR / 'norm_comparison.png'}")
    print(f"Saved position 3 component plot: {OUTPUT_DIR / 'position3_component_comparison.png'}")


if __name__ == "__main__":
    main()
