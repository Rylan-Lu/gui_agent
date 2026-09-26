from __future__ import annotations

from dataclasses import dataclass

from gui_agent.agent.observation import Observation
from gui_agent.locator.ui_locator import normalize_text

def compact_text(text: str) -> str:
    normalized = normalize_text(text)

    return "".join(
        char
        for char in normalized
        if char.isalnum()
    )


@dataclass(frozen=True)
class SuccessJudgeResult:
    success: bool
    reason: str
    matched_text: str | None = None


class TextSuccessJudge:
    def judge(
        self,
        observation: Observation,
        expected_text: str,
    ) -> SuccessJudgeResult:

        if not isinstance(expected_text, str) or not expected_text.strip():
            raise ValueError("expected_text must not be empty")

        if not observation.ocr_result:
            return SuccessJudgeResult(
                success=False,
                reason="no OCR results available",
            )

        target = normalize_text(expected_text)
        compact_target = compact_text(expected_text)

        texts = [
            result.text
            for result in observation.ocr_result
            if result.text.strip()
        ]

        # 普通匹配
        for text in texts:
            candidate = normalize_text(text)

            if target in candidate:
                return SuccessJudgeResult(
                    success=True,
                    reason="expected text found",
                    matched_text=text,
                )

        # OCR 可能拆成多个 block
        combined_text = " ".join(texts)
        combined = normalize_text(combined_text)

        if target in combined:
            return SuccessJudgeResult(
                success=True,
                reason="expected text found across OCR results",
                matched_text=combined_text,
            )

        # OCR 经常丢失 _, -, 空格等符号
        compact_combined = compact_text(combined_text)

        if (
            compact_target
            and compact_target in compact_combined
        ):
            return SuccessJudgeResult(
                success=True,
                reason="expected text found after compact normalization",
                matched_text=combined_text,
            )

        return SuccessJudgeResult(
            success=False,
            reason=f"expected text not found: {expected_text!r}",
        )

class TextAbsentJudge:
    def judge(
        self,
        observation: Observation,
        forbidden_text: str,
    ) -> SuccessJudgeResult:
        if not isinstance(forbidden_text, str) or not forbidden_text.strip():
            raise ValueError("forbidden_text must not be empty")

        if not observation.ocr_result:
            return SuccessJudgeResult(
                success=True,
                reason="forbidden text is absent",
            )

        target = normalize_text(forbidden_text)

        texts = [
            result.text
            for result in observation.ocr_result
            if result.text.strip()
        ]

        combined = normalize_text(" ".join(texts))

        if target in combined:
            return SuccessJudgeResult(
                success=False,
                reason=f"forbidden text still visible: {forbidden_text!r}",
                matched_text=forbidden_text,
            )

        return SuccessJudgeResult(
            success=True,
            reason="forbidden text is absent",
        )