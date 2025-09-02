from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import re

DEFAULT_STOPWORDS = {
	"the", "a", "an", "and", "or", "in", "on", "with", "of", "to", "from", "by",
	"this", "that", "these", "those", "is", "are", "was", "were", "be", "being", "been",
	"for", "at", "as", "it", "its", "into", "over", "under", "near", "very",
}

FOCUS_KEYWORDS = {
	"objects": {"person", "man", "woman", "child", "car", "truck", "bus", "bicycle", "dog", "cat", "table", "chair", "sofa", "bed", "phone", "laptop", "computer", "camera", "ball", "bottle", "cup", "book", "bag"},
	"actions": {"running", "walking", "jumping", "talking", "driving", "eating", "drinking", "cooking", "playing", "working"},
	"people": {"person", "man", "woman", "child", "group", "crowd"},
}


@dataclass
class DistilledTags:
	tags: List[Tuple[str, float]]


class KeywordDistiller:
	def __init__(self, config: Dict) -> None:
		self.config = config
		self.focus: Sequence[str] = config.get("distillation", {}).get("focus", ["objects", "actions", "people"]) or []
		self.stopwords = set(config.get("distillation", {}).get("stopwords") or DEFAULT_STOPWORDS)
		self.window_size = int(config.get("distillation", {}).get("window_size", 5))
		self.stride = int(config.get("distillation", {}).get("stride", 1))

	def extract_keywords(self, caption: str) -> List[str]:
		words = re.findall(r"[a-zA-Z]+", caption.lower())
		keywords = [w for w in words if w not in self.stopwords and len(w) > 2]
		return keywords

	def focus_filter(self, words: List[str]) -> List[str]:
		if not self.focus:
			return words
		allowed: set = set()
		for f in self.focus:
			allowed |= FOCUS_KEYWORDS.get(f, set())
		return [w for w in words if w in allowed]

	def distill(self, captions: Iterable[str], confidences: Iterable[List[Tuple[str, float]]]) -> List[DistilledTags]:
		results: List[DistilledTags] = []
		for caption, tag_scores in zip(captions, confidences):
			if caption:
				words = self.extract_keywords(caption)
				words = self.focus_filter(words)
				# Assign nominal confidence 0.8 for extracted words
				if words:
					uniq = list(dict.fromkeys(words))
					results.append(DistilledTags(tags=[(w, 0.8) for w in uniq]))
					continue
			# Fallback to model tag scores
			results.append(DistilledTags(tags=tag_scores))
		return results