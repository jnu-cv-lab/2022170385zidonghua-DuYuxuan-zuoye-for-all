from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

LAB13_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB13_ROOT))

from utils import LABELS, scan_labeled_videos, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 badminton_storke_video 数据集结构。")
    parser.add_argument("--raw_dir", type=Path, default=LAB13_ROOT / "data" / "raw")
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    out_dir = args.out_dir
    docs_dir = Path(__file__).resolve().parent / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    rows, unmapped = scan_labeled_videos(raw_dir)
    counts = Counter(row["label_name"] for row in rows)
    source_folders = defaultdict(set)
    for row in rows:
        source_folders[row["label_name"]].add(str(row["source_folder"]))

    csv_rows = []
    for label_id, label_name in enumerate(LABELS):
        csv_rows.append(
            {
                "label_id": label_id,
                "label_name": label_name,
                "video_count": counts.get(label_name, 0),
                "source_folders": "; ".join(sorted(source_folders[label_name])),
                "status": "OK" if counts.get(label_name, 0) > 0 else "MISSING",
            }
        )
    write_csv(
        out_dir / "dataset_overview.csv",
        ["label_id", "label_name", "video_count", "source_folders", "status"],
        csv_rows,
    )

    missing_labels = [label for label in LABELS if counts.get(label, 0) == 0]
    lines = [
        "Lab13 数据集检查结果",
        "=" * 24,
        f"raw_dir: {raw_dir}",
        f"已映射视频总数: {len(rows)}",
        f"未映射视频总数: {len(unmapped)}",
        "",
        "六类动作视频数量:",
    ]
    for row in csv_rows:
        lines.append(
            f"- {row['label_id']}: {row['label_name']} -> {row['video_count']} 个视频 "
            f"(来源文件夹: {row['source_folders'] or '未找到'})"
        )
    if missing_labels:
        lines += ["", "缺失类别提示:"]
        lines += [f"- {label}" for label in missing_labels]
    if unmapped:
        lines += ["", "未能映射到六类标签的视频示例:"]
        lines += [f"- {path}" for path in unmapped[:20]]
    (out_dir / "dataset_overview.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    notes = [
        "# Task131 数据集检查说明",
        "",
        "## 数据集来源",
        "本实验使用 Kaggle 数据集 badminton_storke_video。原页面拼写为 storke，本实验按 stroke 击球动作理解。",
        "",
        "## 六类动作名称",
    ]
    notes += [f"- {index}: {label}" for index, label in enumerate(LABELS)]
    notes += [
        "",
        "## 视频数量统计",
    ]
    notes += [f"- {row['label_name']}: {row['video_count']} 个" for row in csv_rows]
    if missing_labels:
        notes += ["", "以下类别当前没有找到视频：" + "、".join(missing_labels)]
    notes += [
        "",
        "## 后续预处理输入路径",
        f"`{raw_dir}`",
        "",
        "检查结果文件：",
        f"- `{out_dir / 'dataset_overview.txt'}`",
        f"- `{out_dir / 'dataset_overview.csv'}`",
    ]
    (docs_dir / "task131_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
