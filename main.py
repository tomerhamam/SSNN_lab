import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from pipeline.config.loader import Config, load_config
from pipeline.extractors.opencv_extractor import OpenCVExtractor, FrameRecord
from pipeline.utils.io import ensure_dir, write_json, write_jsonl, read_jsonl
from pipeline.utils.logging import setup_logger
from pipeline.captioners.clip_zeroshot import ClipZeroShotCaptioner
from pipeline.distillers.keyword import KeywordDistiller
from pipeline.mergers.exact_match import ExactMatchMerger


def run_pipeline(video_path: str, config_path: str | None, overwrite: bool) -> int:
	logger = setup_logger("pipeline")
	cfg = load_config(config_path).to_dict()

	video_path = str(video_path)
	video_stem = Path(video_path).stem
	artifacts_root = ensure_dir(Path(cfg["runtime"]["artifacts_dir"]) / video_stem)
	outputs_root = ensure_dir(Path(cfg["runtime"]["output_dir"]))

	frames_jsonl = artifacts_root / "frames.jsonl"
	captions_jsonl = artifacts_root / "captions.jsonl"
	tags_jsonl = artifacts_root / "tags.jsonl"
	segments_json = outputs_root / f"{video_stem}_segments.json"

	# Stage 1: Extraction
	if overwrite or not frames_jsonl.exists():
		logger.info("[Stage 1] Extracting frames...")
		extractor = OpenCVExtractor(
			target_fps=int(cfg["extraction"]["fps"]),
			resize_hw=tuple(cfg["extraction"]["resize"]) if cfg["extraction"]["resize"] else None,
			max_frames=cfg["extraction"].get("max_frames"),
			save_dir=str(artifacts_root / "frames"),
			save_images=True,
		)
		frames_records: List[FrameRecord] = []
		for rec, _ in extractor.extract(video_path):
			frames_records.append(rec)
		write_jsonl([rec.__dict__ for rec in frames_records], frames_jsonl)
		logger.info(f"Extracted {len(frames_records)} frames -> {frames_jsonl}")
	else:
		logger.info("[Stage 1] Using cached frames")

	# Stage 2: Captioning (CLIP zero-shot)
	if overwrite or not captions_jsonl.exists():
		logger.info("[Stage 2] Captioning frames with CLIP zero-shot...")
		records = read_jsonl(frames_jsonl)
		# Load images from saved paths in small batches
		captioner = ClipZeroShotCaptioner(cfg)
		batch: List[np.ndarray] = []
		batch_ids: List[int] = []
		captured: List[dict] = []
		for rec in records:
			img = None
			if rec.get("saved_path"):
				import cv2  # local import
				img = cv2.imread(rec["saved_path"])
				if img is None:
					continue
			batch.append(img)
			batch_ids.append(rec["frame_id"])
			if len(batch) >= 16:
				res = captioner.predict(batch)
				for fid, r in zip(batch_ids, res):
					captured.append({"frame_id": fid, "caption": r.caption, "tags": r.tags})
				batch, batch_ids = [], []
		# flush
		if batch:
			res = captioner.predict(batch)
			for fid, r in zip(batch_ids, res):
				captured.append({"frame_id": fid, "caption": r.caption, "tags": r.tags})
		write_jsonl(captured, captions_jsonl)
		logger.info(f"Captioned {len(captured)} frames -> {captions_jsonl}")
	else:
		logger.info("[Stage 2] Using cached captions")

	# Stage 3: Distillation (keywords)
	if overwrite or not tags_jsonl.exists():
		logger.info("[Stage 3] Distilling captions to tags...")
		records = {r["frame_id"]: r for r in read_jsonl(captions_jsonl)}
		ordered = [records[k] for k in sorted(records.keys())]
		captions = [r.get("caption", "") for r in ordered]
		model_tags = [r.get("tags", []) for r in ordered]
		distiller = KeywordDistiller(cfg)
		distilled = distiller.distill(captions, model_tags)
		rows = []
		for rec, dist in zip(ordered, distilled):
			rows.append({"frame_id": rec["frame_id"], "tags": dist.tags})
		write_jsonl(rows, tags_jsonl)
		logger.info(f"Distilled tags for {len(rows)} frames -> {tags_jsonl}")
	else:
		logger.info("[Stage 3] Using cached tags")

	# Stage 4: Temporal merging
	logger.info("[Stage 4] Merging into temporal segments...")
	frames = {r["frame_id"]: r for r in read_jsonl(frames_jsonl)}
	tags = {r["frame_id"]: r for r in read_jsonl(tags_jsonl)}
	ordered_ids = sorted(frames.keys())
	timestamps = [frames[i]["timestamp_sec"] for i in ordered_ids]
	tag_lists = [tags[i]["tags"] for i in ordered_ids]
	merger = ExactMatchMerger(min_duration_sec=float(cfg["merging"]["min_duration"]))
	segments = merger.merge(timestamps, tag_lists)
	write_json({
		"video": video_path,
		"segments": [s.__dict__ for s in segments],
	}, segments_json)
	logger.info(f"Wrote segments -> {segments_json}")
	return 0


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Video annotation pipeline")
	parser.add_argument("--video", type=str, required=True, help="Path to input video file")
	parser.add_argument("--config", type=str, default=None, help="Path to config YAML/JSON")
	parser.add_argument("--overwrite", action="store_true", help="Overwrite intermediate artifacts")
	args = parser.parse_args()

	exit(run_pipeline(args.video, args.config, args.overwrite))