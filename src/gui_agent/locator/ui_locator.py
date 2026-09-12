import unicodedata
from dataclasses import dataclass

from gui_agent.ocr.base import OCRResult

class TargetNotFoundError(Exception):
    pass

class AmbiguousTargetError(Exception):
    pass

@dataclass(frozen=True)
class LocatedElement:
    result: OCRResult
    image_center: tuple[float, float]

def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)

    normalized = normalized.casefold()

    normalized = " ".join(normalized.split())

    return normalized

class UILocator:
    def find_all(self, results: list[OCRResult], text: str) -> list[LocatedElement]:
        target = normalize_text(text)

        matches: list[LocatedElement] = []

        for result in results:
            candidate = normalize_text(result.text)

            if candidate != target:
                continue

            center = self._calculate_center(result)

            matches.append(LocatedElement(result, center))

        return matches

    def find_one(self, results: list[OCRResult], text: str) -> LocatedElement:
        matches = self.find_all(results, text)

        if not matches:
            raise TargetNotFoundError(f"Target {text} not found")

        if len(matches) > 1:
            raise AmbiguousTargetError(f"Multiple targets found for {text}: {len(matches)}")

        return matches[0]

    @staticmethod
    def _calculate_center(result: OCRResult) -> tuple[float, float]:
        xs = [point[0] for point in result.bbox]
        ys = [point[1] for point in result.bbox]

        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)

        return center_x, center_y
