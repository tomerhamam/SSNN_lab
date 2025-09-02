import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from pipeline.utils.logging import setup_logger
from pipeline.utils.io import ensure_dir


MODEL_REGISTRY: Dict[str, Dict] = {
	"clip-vit-b-32": {
		"repo": "openai/clip-vit-base-patch32",
		"cache_subdir": "clip",
		"vram_required": 4,
		"capabilities": ["image_caption", "image_embedding"],
	},
	"blip2-base": {
		"repo": "Salesforce/blip2-opt-2.7b",
		"cache_subdir": "blip2",
		"vram_required": 8,
		"capabilities": ["image_caption", "vqa"],
	},
	"llava-1.5-7b": {
		"repo": "liuhaotian/llava-v1.5-7b",
		"cache_subdir": "llava",
		"vram_required": 14,
		"capabilities": ["image_caption", "detailed_description"],
	},
}


class ModelNotFoundError(Exception):
	pass


@dataclass
class LoadedModel:
	name: str
	device: torch.device
	model: object
	processor: Optional[object]


def get_available_vram_gb() -> float:
	try:
		if torch.cuda.is_available():
			props = torch.cuda.get_device_properties(0)
			return float(props.total_memory) / (1024 ** 3)
	except Exception:
		pass
	# Fallback to nvidia-smi
	try:
		result = subprocess.run(
			["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			check=True,
			text=True,
		)
		line = result.stdout.strip().splitlines()[0]
		return float(line) / 1024.0
	except Exception:
		return 0.0


def auto_select_model() -> str:
	vram = get_available_vram_gb()
	if vram < 4:
		return "clip-vit-b-32"  # CPU OK as well
	elif vram < 8:
		return "clip-vit-b-32"
	elif vram < 12:
		return "blip2-base"
	else:
		return "llava-1.5-7b"


class ModelManager:
	def __init__(self, cache_dir: str = "~/.cache/video_pipeline", verify_checksums: bool = True) -> None:
		self.cache_dir = Path(os.path.expanduser(cache_dir))
		self.verify_checksums = verify_checksums
		self.logger = setup_logger("ModelManager")
		ensure_dir(self.cache_dir)

	def get_local_dir(self, model_name: str) -> Path:
		info = MODEL_REGISTRY.get(model_name)
		if info is None:
			raise ModelNotFoundError(f"Unknown model: {model_name}")
		return ensure_dir(self.cache_dir / info["cache_subdir"] / model_name)

	def is_cached(self, model_name: str) -> bool:
		local_dir = self.get_local_dir(model_name)
		return local_dir.exists() and any(local_dir.iterdir())

	def download(self, model_name: str) -> Path:
		info = MODEL_REGISTRY.get(model_name)
		if info is None:
			raise ModelNotFoundError(f"Unknown model: {model_name}")
		local_dir = self.get_local_dir(model_name)
		self.logger.info(f"Downloading model '{model_name}' to {local_dir}")
		path = snapshot_download(
			repo_id=info["repo"],
			local_dir=str(local_dir),
			local_dir_use_symlinks=False,
			resume_download=True,
			max_workers=8,
		)
		return Path(path)

	def ensure_downloaded(self, model_name: str) -> Path:
		if not self.is_cached(model_name):
			return self.download(model_name)
		return self.get_local_dir(model_name)

	def recommend(self, vram_gb: Optional[float] = None) -> List[Tuple[str, int]]:
		if vram_gb is None:
			vram_gb = get_available_vram_gb()
		recs: List[Tuple[str, int]] = []
		for name, meta in MODEL_REGISTRY.items():
			if vram_gb >= float(meta["vram_required"]):
				recs.append((name, meta["vram_required"]))
		recs.sort(key=lambda x: x[1])
		return recs

	def load_clip(self, model_dir: Path, device: torch.device) -> LoadedModel:
		# Use Transformers CLIP
		model = CLIPModel.from_pretrained(str(model_dir))
		processor = CLIPProcessor.from_pretrained(str(model_dir))
		model.to(device)
		model.eval()
		return LoadedModel(name="clip-vit-b-32", device=device, model=model, processor=processor)

	def load(self, model_name: str, device: Optional[torch.device] = None) -> LoadedModel:
		if device is None:
			device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		local_dir = self.ensure_downloaded(model_name)
		if model_name.startswith("clip-"):
			return self.load_clip(local_dir, device)
		# Placeholders for future models
		raise ModelNotFoundError(f"Loading not implemented for model: {model_name}")


class ModelLoader:
	def __init__(self, config: Dict) -> None:
		models_cfg = config.get("models", {})
		self.primary = models_cfg.get("primary", "auto")
		self.fallback = models_cfg.get("fallback", "clip-vit-b-32")
		self.auto_select = bool(models_cfg.get("auto_select", True))
		self.cache_dir = models_cfg.get("cache_dir", "~/.cache/video_pipeline")
		self.download_on_start = bool(models_cfg.get("download_on_start", True))
		self.verify_checksums = bool(models_cfg.get("verify_checksums", True))
		self.manager = ModelManager(cache_dir=self.cache_dir, verify_checksums=self.verify_checksums)
		self.logger = setup_logger("ModelLoader")

	def choose_model(self) -> str:
		if self.primary == "auto" or self.auto_select:
			return auto_select_model()
		return self.primary

	def load(self) -> LoadedModel:
		chosen = self.choose_model()
		try:
			if self.download_on_start:
				self.manager.ensure_downloaded(chosen)
			self.logger.info(f"Loading primary model: {chosen}")
			return self.manager.load(chosen)
		except (torch.cuda.OutOfMemoryError, RuntimeError, ModelNotFoundError) as e:
			self.logger.warning(f"Primary model '{chosen}' failed: {e}. Falling back to '{self.fallback}'.")
			self.manager.ensure_downloaded(self.fallback)
			return self.manager.load(self.fallback)

	def test_model_loading(self, model_name: Optional[str] = None) -> bool:
		name = model_name or self.choose_model()
		loaded = self.manager.load(name)
		# Simple forward pass on dummy image for CLIP
		if name.startswith("clip-"):
			img = Image.new("RGB", (224, 224), color=(128, 128, 128))
			inputs = loaded.processor(images=img, return_tensors="pt").to(loaded.device)
			with torch.no_grad():
				_ = loaded.model.get_image_features(**inputs)
			return True
		return True