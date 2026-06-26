"""Verify the relative-position property of RoPE with NumPy."""

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
    """Return one RoPE frequency for each adjacent 2D dimension group."""
    if d_model % 2 != 0:
        raise ValueError("d_model must be even.")

    group_indices = np.arange(d_model // 2)
    return 1.0 / np.power(10000.0, (2 * group_indices) / d_model)


def apply_rope_to_vector(x: np.ndarray, pos: int) -> np.ndarray:
    """Apply RoPE rotation to a one-dimensional vector at position pos."""
    vector = np.asarray(x, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("x must be a one-dimensional vector.")
    if vector.shape[0] % 2 != 0:
        raise ValueError("d_model must be even.")

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


def rope_dot_score(q: np.ndarray, k: np.ndarray, m: int, n: int) -> float:
    """Return dot(RoPE(q, m), RoPE(k, n))."""
    q_rotated = apply_rope_to_vector(q, m)
    k_rotated = apply_rope_to_vector(k, n)
    return float(np.dot(q_rotated, k_rotated))


def relative_dot_score(q: np.ndarray, k: np.ndarray, relative_pos: int) -> float:
    """Return dot(q, RoPE(k, relative_pos))."""
    k_relative = apply_rope_to_vector(k, relative_pos)
    return float(np.dot(q, k_relative))


def save_heatmap(
    matrix: np.ndarray,
    output_path: Path,
    title: str,
    colorbar_label: str,
) -> None:
    """Save a heatmap for a square position-pair matrix."""
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, aspect="auto", cmap="coolwarm")
    plt.colorbar(label=colorbar_label)
    plt.xlabel("Key position n")
    plt.ylabel("Query position m")
    plt.title(title)
    plt.xticks(np.arange(matrix.shape[1]))
    plt.yticks(np.arange(matrix.shape[0]))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_score_by_relative_position_plot(summary: np.ndarray, output_path: Path) -> None:
    """Save a plot of mean score grouped by relative position."""
    plt.figure(figsize=(7, 4))
    plt.plot(summary[:, 0], summary[:, 1], marker="o")
    plt.xlabel("Relative position n-m")
    plt.ylabel("Mean score")
    plt.title("Mean RoPE Attention Score by Relative Position")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_direct_vs_relative_plot(
    direct_scores: np.ndarray,
    relative_scores: np.ndarray,
    output_path: Path,
) -> None:
    """Save a scatter plot comparing direct and relative scores."""
    min_value = min(float(np.min(direct_scores)), float(np.min(relative_scores)))
    max_value = max(float(np.max(direct_scores)), float(np.max(relative_scores)))
    padding = 0.05 * (max_value - min_value if max_value > min_value else 1.0)
    line_min = min_value - padding
    line_max = max_value + padding

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(direct_scores, relative_scores, alpha=0.8, label="score pairs")
    plt.plot([line_min, line_max], [line_min, line_max], "r--", label="y=x")
    plt.xlabel("direct_score")
    plt.ylabel("relative_score")
    plt.title("Direct Score vs Relative Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def summarize_by_relative_position(results: list[dict[str, float]]) -> np.ndarray:
    """Return columns: relative_position, mean_score, count."""
    relative_positions = sorted({int(result["relative_position"]) for result in results})
    summary_rows = []
    for relative_position in relative_positions:
        scores = [
            result["direct_score"]
            for result in results
            if int(result["relative_position"]) == relative_position
        ]
        summary_rows.append([relative_position, float(np.mean(scores)), len(scores)])
    return np.asarray(summary_rows, dtype=np.float64)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(RANDOM_SEED)
    q = np.random.randn(D_MODEL)
    k = np.random.randn(D_MODEL)

    score_matrix = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float64)
    error_matrix = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float64)
    results = []

    for m in range(SEQ_LEN):
        for n in range(SEQ_LEN):
            relative_position = n - m
            direct_score = rope_dot_score(q, k, m, n)
            relative_score = relative_dot_score(q, k, relative_position)
            error = abs(direct_score - relative_score)

            score_matrix[m, n] = direct_score
            error_matrix[m, n] = error
            results.append(
                {
                    "m": m,
                    "n": n,
                    "relative_position": relative_position,
                    "direct_score": direct_score,
                    "relative_score": relative_score,
                    "error": error,
                }
            )

    errors = error_matrix.reshape(-1)
    max_error = float(np.max(errors))
    mean_error = float(np.mean(errors))
    score_by_relative_position = summarize_by_relative_position(results)

    print(f"q shape: {q.shape}")
    print(f"k shape: {k.shape}")
    print(f"score matrix shape: {score_matrix.shape}")
    print(f"max_error: {max_error:.16e}")
    print(f"mean_error: {mean_error:.16e}")
    print(
        "Verification conclusion: "
        f"{max_error < 1e-10}. "
        "If max_error is very small, RoPE dot products satisfy the relative-position property."
    )

    result_rows = np.asarray(
        [
            [
                result["m"],
                result["n"],
                result["relative_position"],
                result["direct_score"],
                result["relative_score"],
                result["error"],
            ]
            for result in results
        ],
        dtype=np.float64,
    )
    np.savetxt(
        OUTPUT_DIR / "rope_relative_position_results.csv",
        result_rows,
        delimiter=",",
        header="m,n,relative_position,direct_score,relative_score,error",
        comments="",
        fmt=["%d", "%d", "%d", "%.12f", "%.12f", "%.16e"],
    )
    np.savetxt(
        OUTPUT_DIR / "rope_attention_score_matrix.csv",
        score_matrix,
        delimiter=",",
        fmt="%.12f",
    )
    np.savetxt(
        OUTPUT_DIR / "rope_relative_error_matrix.csv",
        error_matrix,
        delimiter=",",
        fmt="%.16e",
    )
    np.savetxt(
        OUTPUT_DIR / "score_by_relative_position.csv",
        score_by_relative_position,
        delimiter=",",
        header="relative_position,mean_score,count",
        comments="",
        fmt=["%d", "%.12f", "%d"],
    )

    save_heatmap(
        score_matrix,
        OUTPUT_DIR / "rope_attention_score_heatmap.png",
        "RoPE Attention Score Heatmap",
        "dot(RoPE(q,m), RoPE(k,n))",
    )
    save_heatmap(
        error_matrix,
        OUTPUT_DIR / "rope_relative_error_heatmap.png",
        "RoPE Relative-Position Error Heatmap",
        "error",
    )
    save_score_by_relative_position_plot(
        score_by_relative_position,
        OUTPUT_DIR / "score_by_relative_position.png",
    )
    save_direct_vs_relative_plot(
        result_rows[:, 3],
        result_rows[:, 4],
        OUTPUT_DIR / "direct_vs_relative_score.png",
    )

    print(f"Saved results: {OUTPUT_DIR / 'rope_relative_position_results.csv'}")
    print(f"Saved score matrix: {OUTPUT_DIR / 'rope_attention_score_matrix.csv'}")
    print(f"Saved error matrix: {OUTPUT_DIR / 'rope_relative_error_matrix.csv'}")
    print(f"Saved score summary: {OUTPUT_DIR / 'score_by_relative_position.csv'}")
    print(f"Saved score heatmap: {OUTPUT_DIR / 'rope_attention_score_heatmap.png'}")
    print(f"Saved error heatmap: {OUTPUT_DIR / 'rope_relative_error_heatmap.png'}")
    print(f"Saved relative-position plot: {OUTPUT_DIR / 'score_by_relative_position.png'}")
    print(f"Saved scatter plot: {OUTPUT_DIR / 'direct_vs_relative_score.png'}")


if __name__ == "__main__":
    main()
