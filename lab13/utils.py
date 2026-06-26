from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


LAB13_ROOT = Path(__file__).resolve().parent

LABELS: List[str] = [
    "forehand drive",
    "forehand lift",
    "forehand net shot",
    "forehand clear",
    "backhand drive",
    "backhand net shot",
]

LABEL_TO_ID: Dict[str, int] = {name: index for index, name in enumerate(LABELS)}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def normalize_label_text(text: str) -> str:
    """Normalize class folder names such as forehand_drive or Forehand Drive."""
    text = text.lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def label_map() -> Dict[str, int]:
    return {name: index for index, name in enumerate(LABELS)}


def save_label_map(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(label_map(), f, ensure_ascii=False, indent=2)


def load_label_map(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(key): int(value) for key, value in data.items()}


def label_id_to_name(mapping: Dict[str, int]) -> Dict[int, str]:
    return {int(value): str(key) for key, value in mapping.items()}


def map_label_from_path(video_path: Path, raw_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """Infer the standard label by checking every folder between raw_dir and file."""
    try:
        relative_parts = video_path.relative_to(raw_dir).parts[:-1]
    except ValueError:
        relative_parts = video_path.parts[:-1]

    normalized_to_label = {normalize_label_text(label): label for label in LABELS}
    for part in reversed(relative_parts):
        normalized = normalize_label_text(part)
        if normalized in normalized_to_label:
            return normalized_to_label[normalized], part
    return None, relative_parts[-1] if relative_parts else None


def find_video_files(raw_dir: Path) -> List[Path]:
    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.exists():
        return []
    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def scan_labeled_videos(raw_dir: Path) -> Tuple[List[Dict[str, object]], List[Path]]:
    rows: List[Dict[str, object]] = []
    unmapped: List[Path] = []
    for video_path in find_video_files(raw_dir):
        label_name, source_folder = map_label_from_path(video_path, raw_dir)
        if label_name is None:
            unmapped.append(video_path)
            continue
        rows.append(
            {
                "video_path": str(video_path),
                "label_name": label_name,
                "label_id": LABEL_TO_ID[label_name],
                "source_folder": source_folder or "",
            }
        )
    rows.sort(key=lambda item: (int(item["label_id"]), str(item["video_path"])))
    return rows, unmapped


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def import_or_explain(module_name: str, package_name: Optional[str] = None):
    package_name = package_name or module_name
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"缺少依赖 {package_name}，请先运行：\n"
            f"pip install -r lab13/requirements_lab13.txt\n"
            f"原始错误：{exc}"
        ) from exc


def resample_sequence(sequence: np.ndarray, target_frames: int) -> np.ndarray:
    if sequence.shape[0] == target_frames:
        return sequence.astype(np.float32)
    if sequence.shape[0] == 0:
        return np.zeros((target_frames, 132), dtype=np.float32)
    indices = np.linspace(0, sequence.shape[0] - 1, target_frames).round().astype(int)
    return sequence[indices].astype(np.float32)


def normalize_landmarks(landmarks: Sequence[object]) -> np.ndarray:
    points = np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
        dtype=np.float32,
    )

    left_shoulder, right_shoulder = points[11, :3], points[12, :3]
    left_hip, right_hip = points[23, :3], points[24, :3]
    hip_center = (left_hip + right_hip) / 2.0
    shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))
    if shoulder_width < 1e-6:
        shoulder_width = 1.0

    points[:, :3] = (points[:, :3] - hip_center) / shoulder_width
    return points.reshape(-1).astype(np.float32)


def extract_skeleton_sequence(
    video_path: Path,
    target_frames: int = 30,
    return_frames: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Extract a [target_frames, 132] MediaPipe Pose skeleton sequence."""
    cv2 = import_or_explain("cv2", "opencv-python")
    mediapipe = import_or_explain("mediapipe")

    mp_pose = mediapipe.solutions.pose
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV 无法打开视频：{video_path}")

    skeleton_frames: List[np.ndarray] = []
    preview_frames: List[np.ndarray] = []
    detected_frames = 0
    total_frames = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            total_frames += 1

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)
            if result.pose_landmarks:
                skeleton_frames.append(normalize_landmarks(result.pose_landmarks.landmark))
                detected_frames += 1
            else:
                skeleton_frames.append(np.zeros(132, dtype=np.float32))

            if return_frames and len(preview_frames) < 12:
                preview_frames.append(frame_bgr.copy())

    capture.release()

    sequence = np.stack(skeleton_frames).astype(np.float32) if skeleton_frames else np.zeros((0, 132), dtype=np.float32)
    resampled = resample_sequence(sequence, target_frames)
    detection_rate = detected_frames / total_frames if total_frames else 0.0
    meta = {
        "video_path": str(video_path),
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "detection_rate": detection_rate,
    }
    if return_frames:
        meta["preview_frames"] = preview_frames
    return resampled, meta


def choose_demo_video(raw_dir: Path) -> Optional[Path]:
    rows, _ = scan_labeled_videos(raw_dir)
    if not rows:
        return None
    return Path(str(rows[0]["video_path"]))


def ensure_project_import() -> None:
    root = str(LAB13_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
