# Video Annotation Pipeline (MVP)

A modular pipeline to generate temporally segmented semantic annotations from RGB videos.

## Features (v1 MVP)
- Frame extraction at configurable FPS (default 1 FPS) with timestamps
- CLIP ViT-B/32 zero-shot captioning over a default label set
- Keyword-based distillation to simple tags
- Exact-match temporal merging with min duration threshold
- Checkpointed stages with JSON/JSONL artifacts
- Model manager with auto-download and hardware-based selection

## Install
```
# Create venv (recommended)
python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

## First-time Setup
```
python setup.py --auto-detect --verify
```
This will auto-select a model based on VRAM, download it to `~/.cache/video_pipeline`, verify loading, and write `config/config.yaml`.

## Run
```
python main.py --video /path/to/video.mp4 --config config/config.yaml --overwrite
```

Outputs:
- Artifacts per video under `artifacts/<video_stem>/`
- Segments JSON at `outputs/<video_stem>_segments.json`

## Configuration Example
```yaml
models:
  auto_select: true
  primary: "auto"
  fallback: "clip-vit-b-32"
  cache_dir: "~/.cache/video_pipeline"
  download_on_start: true
  verify_checksums: true

extraction:
  fps: 1
  resize: [224, 224]

distillation:
  method: "keyword"
  focus: ["objects", "actions", "people"]

merging:
  min_duration: 2.0
  similarity_threshold: 0.8
```

## Notes
- CPU fallback is supported for CLIP; performance is slower.
- For longer videos, artifacts are written incrementally; resume by omitting `--overwrite`.