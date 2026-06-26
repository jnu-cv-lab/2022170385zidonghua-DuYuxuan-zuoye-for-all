from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
from sklearn.model_selection import train_test_split

LAB13_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB13_ROOT))

from utils import LABELS, extract_skeleton_sequence, import_or_explain, save_label_map, scan_labeled_videos, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 MediaPipe Pose 预处理羽毛球动作骨架序列。")
    parser.add_argument("--raw_dir", type=Path, default=LAB13_ROOT / "data" / "raw")
    parser.add_argument("--out_dir", type=Path, default=LAB13_ROOT / "data" / "processed")
    parser.add_argument("--target_frames", type=int, default=30)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--max_videos_per_class", type=int, default=None)
    parser.add_argument("--demo_fast", action="store_true", help="每类最多处理 5 个视频，方便课堂快速跑通。")
    return parser.parse_args()


def limit_videos(rows: List[Dict[str, object]], max_per_class: int | None) -> List[Dict[str, object]]:
    if max_per_class is None:
        return rows
    selected: List[Dict[str, object]] = []
    counts = defaultdict(int)
    for row in rows:
        label = str(row["label_name"])
        if counts[label] < max_per_class:
            selected.append(row)
            counts[label] += 1
    return selected


def can_use_stratify(y: np.ndarray, test_size: float) -> bool:
    unique_labels, counts = np.unique(y, return_counts=True)
    if len(unique_labels) < 2:
        return False
    if counts.min() < 2:
        return False
    test_count = int(round(len(y) * test_size))
    train_count = len(y) - test_count
    return test_count >= len(unique_labels) and train_count >= len(unique_labels)


def write_notes(
    docs_path: Path,
    target_frames: int,
    train_count: int,
    test_count: int,
    output_files: List[Path],
) -> None:
    lines = [
        "# Task132 MediaPipe 骨架预处理说明",
        "",
        "## MediaPipe Pose 提取方法",
        "程序使用 OpenCV 逐帧读取视频，将 BGR 图像转为 RGB 后输入 MediaPipe Pose，提取每帧人体 33 个关键点。",
        "",
        "## 33×4=132 维含义",
        "每个关键点包含 x、y、z、visibility 四个数值，因此单帧骨架特征为 33×4=132 维。",
        "",
        f"## [{target_frames},132] 的含义",
        f"每个视频被重采样为 {target_frames} 帧，每帧 132 维，最终单个视频样本形状为 [{target_frames},132]。",
        "",
        "## 归一化方法",
        "程序以左右髋部中心作为坐标原点，以左右肩部距离作为尺度，对 x、y、z 坐标做平移和缩放；visibility 保持原值。肩宽接近 0 时使用 1.0 避免除零。",
        "",
        "## 训练集/测试集数量",
        f"- 训练集: {train_count}",
        f"- 测试集: {test_count}",
        "",
        "## 输出文件清单",
    ]
    lines += [f"- `{path}`" for path in output_files]
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    out_dir = args.out_dir
    docs_dir = Path(__file__).resolve().parent / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    max_per_class = args.max_videos_per_class
    if args.demo_fast and max_per_class is None:
        max_per_class = 5

    import_or_explain("cv2", "opencv-python")
    import_or_explain("mediapipe")

    rows, unmapped = scan_labeled_videos(raw_dir)
    rows = limit_videos(rows, max_per_class)
    if not rows:
        raise RuntimeError(f"没有在 {raw_dir} 下找到可映射到六类动作的视频。")

    print(f"将处理 {len(rows)} 个视频。raw_dir={raw_dir}")
    X_items: List[np.ndarray] = []
    y_items: List[int] = []
    log_rows: List[Dict[str, object]] = []

    for index, row in enumerate(rows, start=1):
        video_path = Path(str(row["video_path"]))
        label_id = int(row["label_id"])
        label_name = str(row["label_name"])
        print(f"[{index}/{len(rows)}] {label_name}: {video_path.name}")
        try:
            sequence, meta = extract_skeleton_sequence(video_path, target_frames=args.target_frames)
            X_items.append(sequence)
            y_items.append(label_id)
            status = "OK"
            error_message = ""
        except Exception as exc:
            meta = {
                "total_frames": 0,
                "detected_frames": 0,
                "detection_rate": 0.0,
            }
            status = "FAILED"
            error_message = str(exc)

        log_rows.append(
            {
                "video_path": str(video_path),
                "label_id": label_id,
                "label_name": label_name,
                "total_frames": meta["total_frames"],
                "detected_frames": meta["detected_frames"],
                "detection_rate": f"{float(meta['detection_rate']):.4f}",
                "status": status,
                "error": error_message,
            }
        )

        if status == "FAILED":
            print(f"处理失败：{error_message}")

    if not X_items:
        write_csv(
            out_dir / "preprocess_log.csv",
            ["video_path", "label_id", "label_name", "total_frames", "detected_frames", "detection_rate", "status", "error"],
            log_rows,
        )
        raise RuntimeError("所有视频预处理失败，未生成 .npy。请先检查依赖和视频路径。")

    X = np.stack(X_items).astype(np.float32)
    y = np.array(y_items, dtype=np.int64)
    stratify = y if can_use_stratify(y, args.test_size) else None
    if stratify is None:
        print("提示：样本数或类别数不足，train_test_split 不使用 stratify。")

    if len(y) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=42,
            stratify=stratify,
        )
    else:
        X_train, y_train = X, y
        X_test = np.empty((0, args.target_frames, 132), dtype=np.float32)
        y_test = np.empty((0,), dtype=np.int64)

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test.npy", y_test)
    save_label_map(out_dir / "label_map.json")
    write_csv(
        out_dir / "preprocess_log.csv",
        ["video_path", "label_id", "label_name", "total_frames", "detected_frames", "detection_rate", "status", "error"],
        log_rows,
    )

    low_detection = [row for row in log_rows if row["status"] == "OK" and float(row["detection_rate"]) < 0.1]
    summary_lines = [
        "Lab13 预处理总结",
        "=" * 20,
        f"raw_dir: {raw_dir}",
        f"target_frames: {args.target_frames}",
        f"成功样本数: {len(X_items)}",
        f"失败样本数: {sum(1 for row in log_rows if row['status'] == 'FAILED')}",
        f"训练集形状: {list(X_train.shape)}",
        f"测试集形状: {list(X_test.shape)}",
        f"低检测率视频数量(<0.1): {len(low_detection)}",
    ]
    if unmapped:
        summary_lines.append(f"未映射视频数量: {len(unmapped)}")
    (out_dir / "preprocess_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    write_notes(
        docs_dir / "task132_notes.md",
        args.target_frames,
        len(y_train),
        len(y_test),
        [
            out_dir / "X_train.npy",
            out_dir / "y_train.npy",
            out_dir / "X_test.npy",
            out_dir / "y_test.npy",
            out_dir / "label_map.json",
            out_dir / "preprocess_log.csv",
            out_dir / "preprocess_summary.txt",
        ],
    )

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
