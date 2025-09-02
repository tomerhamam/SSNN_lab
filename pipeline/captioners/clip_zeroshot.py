from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from models.model_manager import ModelLoader, get_available_vram_gb


DEFAULT_LABELS: List[str] = [
	"person", "man", "woman", "child", "crowd",
	"car", "truck", "bus", "bicycle", "motorcycle",
	"dog", "cat", "horse", "bird", "cow",
	"table", "chair", "sofa", "bed", "desk",
	"phone", "laptop", "computer", "camera",
	"ball", "bottle", "cup", "book", "bag",
	"running", "walking", "jumping", "talking", "driving",
	"eating", "drinking", "cooking", "playing", "working",
]


@dataclass
class CaptionResult:
	caption: str
	tags: List[Tuple[str, float]]


class ClipZeroShotCaptioner:
	def __init__(
		self,
		config: dict,
		candidate_labels: Optional[Sequence[str]] = None,
		top_k: int = 5,
	) -> None:
		self.config = config
		self.labels = list(candidate_labels) if candidate_labels else DEFAULT_LABELS
		self.top_k = int(top_k)
		batch_size_cfg = config.get("captioning", {}).get("batch_size", "auto")
		if batch_size_cfg == "auto":
			vram = get_available_vram_gb()
			self.batch_size = 8 if vram < 4 else 16
		else:
			self.batch_size = int(batch_size_cfg)
		self.model_loader = ModelLoader(config)
		self.loaded = self.model_loader.load()

	def _score_batch(self, images: List[Image.Image]) -> List[CaptionResult]:
		processor = self.loaded.processor
		model = self.loaded.model
		device = self.loaded.device
		# Prepare text prompts
		texts = [f"a photo of {label}" for label in self.labels]
		text_inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
		with torch.no_grad():
			text_features = model.get_text_features(**text_inputs)
			text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

		results: List[CaptionResult] = []
		for i in range(0, len(images), self.batch_size):
			batch = images[i:i + self.batch_size]
			inputs = processor(images=batch, return_tensors="pt").to(device)
			with torch.no_grad():
				image_features = model.get_image_features(**inputs)
				image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
				scores = image_features @ text_features.T  # cosine similarity since both normalized
				scores = scores.softmax(dim=-1)

			for row in scores:
				row_np = row.detach().cpu().numpy()
				top_idx = np.argsort(row_np)[-self.top_k:][::-1]
				top = [(self.labels[j], float(row_np[j])) for j in top_idx]
				caption = ", ".join([lbl for lbl, _ in top])
				results.append(CaptionResult(caption=caption, tags=top))
		return results

	def predict(self, images_bgr: List[np.ndarray]) -> List[CaptionResult]:
		# Convert BGR to RGB PIL
		pil_images = [Image.fromarray(img[..., ::-1]) for img in images_bgr]
		return self._score_batch(pil_images)