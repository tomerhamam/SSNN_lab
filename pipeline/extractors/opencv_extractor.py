from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple

import cv2
import numpy as np

from pipeline.utils.io import ensure_dir


@dataclass
class FrameRecord:
	video_path: str
	frame_id: int
	timestamp_sec: float
	width: int
	height: int
	saved_path: Optional[str] = None


class OpenCVExtractor:
	def __init__(
		self,
		target_fps: int = 1,
		resize_hw: Optional[Tuple[int, int]] = (224, 224),
		max_frames: Optional[int] = None,
		save_dir: Optional[str] = None,
		save_images: bool = False,
		jpeg_quality: int = 90,
	) -> None:
		self.target_fps = max(1, int(target_fps))
		self.resize_hw = resize_hw
		self.max_frames = max_frames
		self.save_dir = Path(save_dir) if save_dir else None
		self.save_images = save_images
		self.jpeg_quality = int(np.clip(jpeg_quality, 10, 100))
		if self.save_images and self.save_dir is not None:
			ensure_dir(self.save_dir)

	def _should_keep(self, frame_idx: int, step: int) -> bool:
		return frame_idx % step == 0

	def _save_frame(self, image_bgr: np.ndarray, dst_path: Path) -> None:
		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
		cv2.imwrite(str(dst_path), image_bgr, encode_param)

	def extract(self, video_path: str) -> Generator[Tuple[FrameRecord, np.ndarray], None, None]:
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			raise RuntimeError(f"Failed to open video: {video_path}")

		video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
		if video_fps <= 0:
			# Fallback to sampling every frame when FPS unknown; timestamps will increment by 1
			video_fps = float(self.target_fps)
			step = 1
		else:
			step = max(int(round(video_fps / float(self.target_fps))), 1)

		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
		video_stem = Path(video_path).stem
		frames_saved_dir: Optional[Path] = None
		if self.save_images and self.save_dir is not None:
			frames_saved_dir = ensure_dir(self.save_dir / f"frames_{video_stem}")

		kept = 0
		frame_idx = 0
		while True:
			ret, frame_bgr = cap.read()
			if not ret:
				break
			if not self._should_keep(frame_idx, step):
				frame_idx += 1
				continue

			height, width = frame_bgr.shape[:2]
			if self.resize_hw is not None:
				res_h, res_w = int(self.resize_hw[0]), int(self.resize_hw[1])
				frame_bgr = cv2.resize(frame_bgr, (res_w, res_h), interpolation=cv2.INTER_AREA)
				height, width = res_h, res_w

			timestamp_sec = float(frame_idx) / float(video_fps) if video_fps > 0 else float(kept)

			saved_path_str: Optional[str] = None
			if frames_saved_dir is not None:
				img_name = f"{video_stem}_f{frame_idx:06d}.jpg"
				dst = frames_saved_dir / img_name
				self._save_frame(frame_bgr, dst)
				saved_path_str = str(dst)

			record = FrameRecord(
				video_path=str(video_path),
				frame_id=frame_idx,
				timestamp_sec=timestamp_sec,
				width=width,
				height=height,
				saved_path=saved_path_str,
			)

			yield record, frame_bgr

			kept += 1
			frame_idx += 1
			if self.max_frames is not None and kept >= self.max_frames:
				break

		cap.release()

	def extract_metadata(self, video_path: str) -> Iterable[FrameRecord]:
		for rec, _ in self.extract(video_path):
			yield rec