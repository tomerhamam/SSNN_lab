import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "type": "clip",
        "variant": "ViT-B/32",
        "device": "auto",
    },
    "extraction": {
        "fps": 1,
        "resize": [224, 224],
        "batch_size": 16,
        "max_frames": None,
    },
    "distillation": {
        "method": "keyword",
        "focus": ["objects", "actions", "people"],
        "stopwords": None,
        "window_size": 5,
        "stride": 1,
    },
    "merging": {
        "min_duration": 2.0,
        "similarity_threshold": 0.8,
    },
    "runtime": {
        "num_workers": 0,
        "output_dir": "outputs",
        "artifacts_dir": "artifacts",
        "overwrite": False,
        "checkpoint": True,
    },
}


class Config:
    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        merged = DEFAULT_CONFIG.copy()
        if data:
            merged = deep_update(merged, data)
        self.data = merged

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.data
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def to_dict(self) -> Dict[str, Any]:
        return self.data


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Optional[str]) -> Config:
    if path is None:
        return Config()
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if cfg_path.suffix.lower() in {".yml", ".yaml"}:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    elif cfg_path.suffix.lower() == ".json":
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Unsupported config format. Use YAML or JSON.")
    return Config(data)