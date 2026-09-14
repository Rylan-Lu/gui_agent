import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal, TypeAlias

from gui_agent.ocr.base import BoundingBox, OCRResult

ROI: TypeAlias = tuple[int, int, int, int]
MatchMode: TypeAlias = Literal["exact", "fuzzy"]


class TargetNotFoundError(Exception):
    pass


class AmbiguousTargetError(Exception):
    pass


@dataclass(frozen=True)
class LocatedElement:
    result: OCRResult
    image_center: tuple[float, float]
    match_score: float

    @property
    def image_bbox(self) -> BoundingBox:
        return self.result.bbox


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = " ".join(normalized.split())

    quote_pairs = {
        ("“", "”"),
        ("‘", "’"),
        ('"', '"'),
        ("'", "'"),
    }

    while len(normalized) >= 2 and (
        normalized[0],
        normalized[-1],
    ) in quote_pairs:
        normalized = normalized[1:-1].strip()

    return normalized


class UILocator:
    def find_all(
        self,
        results: list[OCRResult],
        text: str,
        *,
        mode: MatchMode = "exact",
        min_confidence: float = 0.0,
        fuzzy_threshold: float = 0.8,
        roi: ROI | None = None,
    ) -> list[LocatedElement]:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if not 0.0 <= fuzzy_threshold <= 1.0:
            raise ValueError("fuzzy_threshold must be between 0.0 and 1.0")

        target = normalize_text(text)
        matches: list[LocatedElement] = []

        for result in results:
            if result.confidence < min_confidence:
                continue

            center = self._calculate_center(result)
            if roi is not None and not self._point_in_roi(center, roi):
                continue

            candidate = normalize_text(result.text)
            match_score = self._calculate_match_score(target, candidate, mode)

            if mode == "exact" and match_score < 1.0:
                continue
            if mode == "fuzzy" and match_score < fuzzy_threshold:
                continue

            matches.append(
                LocatedElement(
                    result=result,
                    image_center=center,
                    match_score=match_score,
                )
            )

        matches.sort(
            key=lambda item: (item.match_score, item.result.confidence),
            reverse=True,
        )
        return matches

    def find_one(
        self,
        results: list[OCRResult],
        text: str,
        *,
        mode: MatchMode = "exact",
        min_confidence: float = 0.0,
        fuzzy_threshold: float = 0.8,
        roi: ROI | None = None,
    ) -> LocatedElement:
        matches = self.find_all(
            results,
            text,
            mode=mode,
            min_confidence=min_confidence,
            fuzzy_threshold=fuzzy_threshold,
            roi=roi,
        )

        if not matches:
            raise TargetNotFoundError(f"Target {text!r} not found")
        if len(matches) > 1:
            raise AmbiguousTargetError(
                f"Multiple targets found for {text!r}: {len(matches)}"
            )

        return matches[0]

    @staticmethod
    def _calculate_center(result: OCRResult) -> tuple[float, float]:
        xs = [point[0] for point in result.bbox]
        ys = [point[1] for point in result.bbox]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    @staticmethod
    def _calculate_match_score(
        target: str,
        candidate: str,
        mode: MatchMode,
    ) -> float:
        if mode == "exact":
            return 1.0 if target == candidate else 0.0
        if mode == "fuzzy":
            return SequenceMatcher(None, target, candidate).ratio()
        raise ValueError(f"Invalid mode: {mode}")

    @staticmethod
    def _point_in_roi(point: tuple[float, float], roi: ROI) -> bool:
        x, y = point
        left, top, width, height = roi

        if width <= 0 or height <= 0:
            raise ValueError("ROI width and height must be positive")

        right = left + width
        bottom = top + height
        return left <= x < right and top <= y < bottom
