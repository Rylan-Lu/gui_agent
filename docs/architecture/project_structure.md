# Project structure and architecture

## Design goal

The repository uses a layered but deliberately lightweight architecture. The objective is to keep perception, model calls, planning, grounding, execution, and evaluation independently testable without introducing a large framework prematurely.

## Product code

### `perception/`

**Responsibility:** acquire the current desktop image.

- `screen_capture.py` — MSS full-screen/region capture and `CapturedFrame` metadata.

Output flows into OCR, feedback, and later VLM observation.

### `ocr/`

**Responsibility:** convert screen pixels into text observations.

- `base.py` — `OCRResult` and `OCREngine` abstraction.
- `paddleocr_engine.py` — primary PaddleOCR adapter; GPU mode enforces Torch-first initialization.
- `easyocr_engine.py` — fallback/debug EasyOCR adapter; GPU can be disabled for CPU-only diagnostics.
- `visualizer.py` — OCR visualization helper.

### `locator/`

**Responsibility:** turn OCR text results into target positions.

- `ui_locator.py` — text normalization, exact/fuzzy match, confidence/ROI filtering, ambiguity handling.
- `coordinate_mapper.py` — image/region coordinates → desktop coordinates.
- `visualizer.py` — localization visualization helper.

### `control/`

**Responsibility:** perform real desktop input.

- `controller.py` — mouse, keyboard, hotkey, scroll, drag operations.
- `windows_text.py` — Windows `SendInput + KEYEVENTF_UNICODE` text input.

This package is currently Windows-oriented because `windows_text.py` depends directly on Win32 APIs.

### `models/`

**Responsibility:** present one common request/response interface for model backends.

- `base.py` — `ModelRequest`, `ModelResponse`, `ModelClient` protocol.
- `local_vlm.py` — local Qwen3-VL client.
- `api_model.py` — OpenAI-compatible HTTPS Chat Completions adapter.

### `datasets/`

**Responsibility:** normalize public GUI datasets into common action/sample schemas.

- `schema.py` — common `GUIAction`, `GUIExample`, coordinate/evaluation types.
- `screenagent.py`, `mind2web.py`, `webarena.py` — dataset-specific adapters.

Dataset schemas are reused by the live agent action path, which avoids defining a second incompatible action representation.

### `agent/`

**Responsibility:** connect planning to desktop execution and task evaluation.

The package currently contains four logical groups:

1. **Planning** — `prompts.py`, `planner.py`, `plan_schema.py`, `chain.py`.
2. **Execution bridge** — `action_adapter.py`, `grounding.py`, `executor.py`, `action_runner.py`.
3. **Runtime/environment** — `runtime.py`, `observation.py`, `environment.py`, `action_transaction.py`.
4. **Evaluation** — `feedback.py`, `success_judge.py`, `loop.py`.

They remain in one package for the v1 baseline to avoid a disruptive import-only refactor. If this package grows substantially, these four groups are the natural future package boundaries.

### `runtime/`

**Responsibility:** process-level runtime prerequisites.

- `gpu_runtime.py` — idempotent Torch CUDA/cuDNN initialization used to preserve the validated Torch-first loading order.

## Runtime flow

```text
Planner
  │
  ▼
TaskPlan / PlanStep
  │
  ▼
AgentLoop
  │
  ▼
plan_step_to_action
  │
  ▼
ActionTransaction
  ├─ before Observation (+ OCR when grounding is required)
  ├─ ActionRunner
  │    ├─ GroundingBridge
  │    │    ├─ UILocator
  │    │    └─ CoordinateMapper
  │    └─ ActionExecutor
  │         └─ Controller
  ├─ settle
  ├─ after Observation
  └─ ActionFeedback
  │
  ▼
Success Judge
  │
  ▼
Done / Failed
```

## Test boundaries

- `tests/` — deterministic automated tests using fakes/monkeypatches; safe for frequent regression runs.
- `scripts/diagnostics/` — real GPU/model/API checks.
- `scripts/e2e/` — tests that can interact with the real desktop.
- `scripts/benchmarks/` — performance experiments.

Keeping these separate prevents a normal `pytest` run from unexpectedly controlling the desktop or loading large models.
