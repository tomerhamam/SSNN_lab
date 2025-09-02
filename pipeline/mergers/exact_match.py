from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class Segment:
	start_time: float
	end_time: float
	tags: List[str]


class ExactMatchMerger:
	def __init__(self, min_duration_sec: float = 2.0):
		self.min_duration_sec = float(min_duration_sec)

	@staticmethod
	def _tag_set(tags: Sequence[Tuple[str, float]]) -> Tuple[str, ...]:
		# Use sorted unique tag names only for exact matching
		names = sorted({t for t, _ in tags})
		return tuple(names)

	def merge(self, timestamps: List[float], tag_lists: List[Sequence[Tuple[str, float]]]) -> List[Segment]:
		if not timestamps:
			return []
		# Assume timestamps aligned to frames at 1/fps; use next timestamp as end; for last frame, extend by delta
		deltas = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
		avg_delta = deltas[0] if deltas else 1.0

		segments: List[Segment] = []
		current_tags = self._tag_set(tag_lists[0])
		seg_start = timestamps[0]
		for i in range(1, len(timestamps)):
			tag_set = self._tag_set(tag_lists[i])
			if tag_set != current_tags:
				seg_end = timestamps[i]
				if seg_end - seg_start >= self.min_duration_sec:
					segments.append(Segment(start_time=seg_start, end_time=seg_end, tags=list(current_tags)))
				seg_start = timestamps[i]
				current_tags = tag_set
		# last segment
		last_end = timestamps[-1] + avg_delta
		if last_end - seg_start >= self.min_duration_sec:
			segments.append(Segment(start_time=seg_start, end_time=last_end, tags=list(current_tags)))
		return segments