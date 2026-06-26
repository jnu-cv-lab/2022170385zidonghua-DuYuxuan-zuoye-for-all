"""Generate sinusoidal position encoding with NumPy."""

import os
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = TASK_DIR / "output"
MPL_CONFIG_DIR = OUTPUT_DIR / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np


SEQ_LEN = 32
D_MODEL = 16


def sinusoidal_position_encoding(seq_len: int = SEQ_LEN, d_model: int = D_MODEL) -> np.ndarray:
    """Create a sinusoidal position encoding matrix of shape (seq_len, d_model)."""
    positions = np.arange(seq_len)[:, np.newaxis]
    even_dimensions = np.arange(0, d_model, 2)
    div_terms = np.power(10000.0, even_dimensions / d_model)

    encoding = np.zeros((seq_len, d_model), dtype=np.float64)
    encoding[:, 0::2] = np.sin(positions / div_terms)
    encoding[:, 1::2] = np.cos(positions / div_terms)
    return encoding


def save_heatmap(encoding: np.ndarray, output_path: Path) -> None:
    """Save a heatmap visualization of the full position encoding matrix."""
    plt.figure(figsize=(10, 5))
    plt.imshow(encoding, aspect="auto", cmap="viridis")
    plt.colorbar(label="Encoding value")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.title("Sinusoidal Position Encoding Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_curves(encoding: np.ndarray, output_path: Path, num_dims: int = 8) -> None:
    """Save curves showing how the first dimensions change with position."""
    positions = np.arange(encoding.shape[0])

    plt.figure(figsize=(10, 5))
    for dim in range(min(num_dims, encoding.shape[1])):
        plt.plot(positions, encoding[:, dim], marker="o", markersize=3, label=f"dim {dim}")

    plt.xlabel("Position")
    plt.ylabel("Encoding value")
    plt.title("Sinusoidal Position Encoding Curves")
    plt.legend(ncol=4, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    encoding = sinusoidal_position_encoding()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "position_encoding_matrix.csv"
    heatmap_path = OUTPUT_DIR / "position_encoding_heatmap.png"
    curves_path = OUTPUT_DIR / "position_encoding_curves.png"

    np.savetxt(csv_path, encoding, delimiter=",", fmt="%.10f")
    save_heatmap(encoding, heatmap_path)
    save_curves(encoding, curves_path)

    print(f"Position encoding matrix shape: {encoding.shape}")
    print("First 5 positions and first 8 dimensions:")
    print(np.array2string(encoding[:5, :8], precision=6, suppress_small=False))
    print("Position 0 encoding:")
    print(np.array2string(encoding[0], precision=6, suppress_small=False))
    print(f"Saved CSV: {csv_path}")
    print(f"Saved heatmap: {heatmap_path}")
    print(f"Saved curves: {curves_path}")


if __name__ == "__main__":
    main()
