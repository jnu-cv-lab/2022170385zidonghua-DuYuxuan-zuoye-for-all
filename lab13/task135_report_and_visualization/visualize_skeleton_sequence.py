from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "output" / ".matplotlib"))

import matplotlib.pyplot as plt

LAB13_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB13_ROOT))

from utils import choose_demo_video, import_or_explain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="保存 MediaPipe Pose 骨架可视化图片。")
    parser.add_argument("--video_path", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--max_drawn_frames", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = Path(__file__).resolve().parent / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    video_path = args.video_path
    if video_path is None:
        video_path = choose_demo_video(LAB13_ROOT / "data" / "raw")
        if video_path is None:
            raise FileNotFoundError("没有找到 demo 视频，请使用 --video_path 指定一个真实视频。")
    video_path = video_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在：{video_path}")

    cv2 = import_or_explain("cv2", "opencv-python")
    mediapipe = import_or_explain("mediapipe")
    mp_pose = mediapipe.solutions.pose
    mp_drawing = mediapipe.solutions.drawing_utils
    mp_styles = mediapipe.solutions.drawing_styles

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV 无法打开视频：{video_path}")

    drawn_frames = []
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while len(drawn_frames) < args.max_drawn_frames:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)
            if not result.pose_landmarks:
                continue
            drawn = frame_bgr.copy()
            mp_drawing.draw_landmarks(
                drawn,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
            )
            drawn_frames.append(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
    capture.release()

    if not drawn_frames:
        raise RuntimeError("未检测到可视化用的人体骨架帧。")

    plt.figure(figsize=(6, 4))
    plt.imshow(drawn_frames[0])
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(args.out_dir / "skeleton_sample.png", dpi=160)
    plt.close()

    columns = min(3, len(drawn_frames))
    rows = (len(drawn_frames) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows))
    axes_list = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, frame in zip(axes_list, drawn_frames):
        ax.imshow(frame)
        ax.axis("off")
    for ax in axes_list[len(drawn_frames) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out_dir / "skeleton_sequence_overview.png", dpi=160)
    plt.close(fig)

    notes = [
        "# Task135 骨架可视化说明",
        "",
        "骨架可视化用于检查 MediaPipe Pose 是否能在羽毛球视频中稳定检测人体关键点。若关键点明显偏移、缺失或整段视频无法检测，将直接影响后续 [30,132] 骨架序列质量和分类准确率。",
        "",
        "输出文件：",
        f"- `{args.out_dir / 'skeleton_sample.png'}`",
        f"- `{args.out_dir / 'skeleton_sequence_overview.png'}`",
    ]
    (docs_dir / "task135_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"已保存骨架可视化到：{args.out_dir}")


if __name__ == "__main__":
    main()
