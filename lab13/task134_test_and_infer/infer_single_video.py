from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch

LAB13_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB13_ROOT))
sys.path.insert(0, str(LAB13_ROOT / "task133_train_transformer"))

from skeleton_transformer import SkeletonTransformer
from utils import LABELS, choose_demo_video, extract_skeleton_sequence, label_id_to_name, load_label_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对单个羽毛球视频进行动作识别推理。")
    parser.add_argument("--video_path", type=Path, default=None)
    parser.add_argument("--processed_dir", type=Path, default=LAB13_ROOT / "data" / "processed")
    parser.add_argument("--model_path", type=Path, default=LAB13_ROOT / "task133_train_transformer" / "output" / "best_model.pth")
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--target_frames", type=int, default=30)
    return parser.parse_args()


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.video_path
    if video_path is None:
        video_path = choose_demo_video(LAB13_ROOT / "data" / "raw")
        if video_path is None:
            raise FileNotFoundError("没有找到 demo 视频，请使用 --video_path 指定一个真实视频。")
    video_path = video_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在：{video_path}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"模型不存在：{args.model_path}")

    mapping = load_label_map(args.processed_dir / "label_map.json")
    id_to_name = label_id_to_name(mapping)
    sequence, meta = extract_skeleton_sequence(video_path, target_frames=args.target_frames)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch_load(args.model_path, device)
    model = SkeletonTransformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        X = torch.tensor(sequence[None, :, :], dtype=torch.float32, device=device)
        probabilities = torch.softmax(model(X), dim=1).cpu().numpy()[0]
    pred_id = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_id])
    pred_name = id_to_name.get(pred_id, LABELS[pred_id])

    lines = [
        "Lab13 单视频推理结果",
        "=" * 24,
        f"video_path: {video_path}",
        f"predicted class: {pred_name}",
        f"confidence: {confidence:.6f}",
        f"total_frames: {meta['total_frames']}",
        f"detected_frames: {meta['detected_frames']}",
        f"detection_rate: {float(meta['detection_rate']):.4f}",
        "",
        "softmax probabilities:",
    ]
    for class_id, prob in enumerate(probabilities):
        lines.append(f"- {class_id}: {id_to_name.get(class_id, LABELS[class_id])}: {float(prob):.6f}")
    (args.out_dir / "inference_result.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (args.out_dir / "inference_probabilities.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "probability"])
        writer.writeheader()
        for class_id, prob in enumerate(probabilities):
            writer.writerow(
                {
                    "class_id": class_id,
                    "class_name": id_to_name.get(class_id, LABELS[class_id]),
                    "probability": f"{float(prob):.6f}",
                }
            )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
