import argparse
from pathlib import Path
from typing import Optional

from pipeline.config.loader import DEFAULT_CONFIG, Config
from pipeline.utils.io import ensure_dir, write_json
from pipeline.utils.logging import setup_logger
from models.model_manager import ModelLoader, ModelManager, auto_select_model, get_available_vram_gb


def build_config(primary: str, fallback: str, cache_dir: str) -> dict:
	cfg = DEFAULT_CONFIG.copy()
	cfg["models"] = cfg.get("models", {})
	cfg["models"].update({
		"auto_select": False,
		"primary": primary,
		"fallback": fallback,
		"cache_dir": cache_dir,
		"download_on_start": True,
		"verify_checksums": True,
	})
	cfg["captioning"] = cfg.get("captioning", {"model": "auto", "batch_size": "auto"})
	cfg["tagging"] = cfg.get("tagging", {"model": "mistral-7b-instruct", "quantization": "4bit"})
	return cfg


def setup_pipeline(model: Optional[str], cache_dir: str, verify: bool) -> int:
	logger = setup_logger("setup")
	manager = ModelManager(cache_dir=cache_dir)
	if model is None or model == "auto":
		recommended = auto_select_model()
		logger.info(f"Auto-selected model based on VRAM: {recommended}")
		selected = recommended
	else:
		selected = model

	# Ensure download
	manager.ensure_downloaded(selected)

	# Verify loading
	if verify:
		loader = ModelLoader({
			"models": {
				"primary": selected,
				"fallback": "clip-vit-b-32",
				"cache_dir": cache_dir,
			}
		})
		ok = loader.test_model_loading(selected)
		if not ok:
			logger.error("Model verification failed.")
			return 2
		logger.info("Model verification succeeded.")

	# Save config
	cfg = build_config(primary=selected, fallback="clip-vit-b-32", cache_dir=cache_dir)
	config_dir = ensure_dir(Path("/workspace/config"))
	out_path = config_dir / "config.yaml"
	try:
		import yaml  # type: ignore
		with open(out_path, "w", encoding="utf-8") as f:
			yaml.safe_dump(cfg, f, sort_keys=False)
	except Exception:
		# Fallback to JSON if YAML unavailable
		write_json(cfg, config_dir / "config.json")
		logger.warning("Could not write YAML config; wrote JSON instead.")
		return 0

	logger.info(f"Wrote configuration to {out_path}")
	return 0


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Initial setup for video annotation pipeline")
	parser.add_argument("--auto-detect", action="store_true", help="Auto-detect GPU and select model")
	parser.add_argument("--model", type=str, default=None, help="Specific model to download (e.g., clip-vit-b-32)")
	parser.add_argument("--cache-dir", type=str, default="~/.cache/video_pipeline", help="Directory to cache models")
	parser.add_argument("--verify", action="store_true", help="Verify model loads correctly after download")
	args = parser.parse_args()

	selected = None
	if args.auto_detect:
		selected = "auto"
	elif args.model:
		selected = args.model

	exit(setup_pipeline(selected, args.cache_dir, args.verify))