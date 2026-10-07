# GUI Agent

A modular Windows desktop GUI agent built around screen perception, OCR/VLM understanding, task planning, grounding, desktop execution, feedback, and success judging.

## Current status

The repository contains the Week 1–4 GUI Agent v1 baseline.

Implemented capabilities include:

- MSS full-screen and region capture.
- PaddleOCR primary OCR backend and EasyOCR fallback/debug backend.
- Exact/fuzzy text localization and coordinate mapping.
- Mouse, keyboard, scrolling, dragging, hotkeys, and IME-safe Unicode text input.
- Local Qwen3-VL and OpenAI-compatible remote model adapters.
- GUI dataset adapters for ScreenAgent, Mind2Web, and WebArena.
- Planner → `TaskPlan` / `PlanStep` structured planning.
- Agent runtime, grounding, execution, screen-change feedback, and success judging.
- Real desktop E2E scripts and a five-task Week 4 baseline suite.

The current implementation is a working engineering baseline, not a finished production agent. Retry/recovery/replan, non-text visual grounding, broader task evaluation, and systematic performance profiling remain active development areas.

## Runtime architecture

```text
User Task
   ↓
Planner
   ↓
TaskPlan / PlanStep
   ↓
AgentLoop
   ↓
GUIAction
   ↓
Grounding (OCR + UILocator + CoordinateMapper)
   ↓
ActionRunner → ActionExecutor → Controller
   ↓
Desktop
   ↓
Observation
   ├─ ActionFeedback
   └─ Success Judge
   ↓
Next / Finish / Fail
```

## Repository layout

```text
gui_agent/
├── src/gui_agent/           # Product/runtime code
│   ├── agent/               # Planner, runtime loop, execution bridge, feedback/judge
│   ├── control/             # Mouse/keyboard and Windows Unicode input
│   ├── datasets/            # ScreenAgent / Mind2Web / WebArena adapters
│   ├── locator/             # Text localization and coordinate mapping
│   ├── models/              # Local VLM and remote API model clients
│   ├── ocr/                 # OCR interfaces/backends
│   ├── perception/          # Screen capture
│   └── runtime/             # GPU runtime initialization
├── tests/                   # Fast automated pytest suite
├── scripts/
│   ├── benchmarks/          # Performance experiments
│   ├── datasets/            # Dataset export/audit/preprocessing
│   ├── diagnostics/         # Real model/API/GPU checks
│   └── e2e/                 # Real desktop end-to-end runs
├── docs/
│   ├── architecture/        # Architecture and structure notes
│   ├── guides/              # Usage/runbooks
│   └── reports/             # Historical development reports
├── artifacts/               # Generated runtime outputs; ignored by Git
├── data/                    # Local datasets; ignored by Git
├── pyproject.toml
├── environment-stable.yml
└── requirements-stable.txt
```

See [`docs/architecture/project_structure.md`](docs/architecture/project_structure.md) for module responsibilities and dependency flow.

## Environment

Validated development baseline:

- Windows 11
- Python 3.11.15
- NVIDIA GPU environment
- PyTorch + PaddleOCR GPU coexistence using **Torch-first initialization**

Important runtime constraints:

1. Initialize PyTorch CUDA/cuDNN before importing Paddle/PaddleOCR in GPU workflows.
2. PaddleOCR is intentionally lazy-imported.
3. Do not apply an additional `1 / 1.25` coordinate correction on the validated Windows 125% DPI setup; MSS and PyAutoGUI were verified in the same coordinate space.
4. API keys must remain in environment variables and must not be committed.
5. Avoid logging complete full-screen OCR output because desktop screenshots may contain sensitive information.

The frozen environment files are historical snapshots. The Paddle GPU setup also depends on a validated Windows-specific wheel/cuDNN arrangement; see the project handoff/environment notes before rebuilding the GPU stack.

## Install for development

From the repository root:

```powershell
conda activate gui-agent
python -m pip install -e ".[dev]"
python -m pip check
```

Optional capabilities are installed separately from `pyproject.toml`, for example:

```powershell
python -m pip install -e ".[paddleocr]"
python -m pip install -e ".[easyocr]"
python -m pip install -e ".[agent]"
python -m pip install -e ".[local-vlm]"
```

For the already validated GPU environment, prefer the existing stable environment over casually reinstalling Torch/Paddle packages.

## Automated tests

```powershell
python -m pytest -q
```

The historical Week 4 full-regression checkpoint reached **427 passed**. Additional Success Judge coverage was added afterward, so establish the current baseline from an actual fresh `pytest` run rather than assuming the historical count.

`tests/` is reserved for automated unit/integration tests that should not manipulate the real desktop. Real desktop/API/GPU checks live under `scripts/`.

## Real desktop checks

Real scripts can move the mouse, type text, open applications, call APIs, or load large GPU models. Run them deliberately.

Examples:

```powershell
python scripts/e2e/test_week4_real_e2e.py
python scripts/e2e/week4_task_suite.py
python scripts/diagnostics/test_gpu_coexistence.py
```

See [`scripts/README.md`](scripts/README.md) for the full classification.

## Performance baseline

Historical stage measurements include:

| Component | Recorded result | Scope |
|---|---:|---|
| PaddleOCR GPU | ~1.919–1.971 s / image | Same desktop screenshot |
| EasyOCR GPU | ~4.170 s / image | Same desktop screenshot |
| Local Qwen3-VL load | ~5.28 s | Small number of runs |
| Local Qwen3-VL inference | ~2–3 s | Input/output length dependent |
| Remote multimodal API | ~1.66–1.7 s | Single-request reference |
| Planner E2E | ~12.16 s | Instruction + screenshot → VLM → JSON → TaskPlan |

These are engineering observations, not a standardized benchmark. Future performance reports should keep raw timing and report at least Mean, P95, and Std under controlled warm-up/device conditions.

## Current limitations

- OCR grounding mainly handles visible text controls.
- Icon-only and other non-text controls need UIA/VLM or hybrid grounding.
- Pixel-change feedback indicates that the screen changed, not that the task succeeded.
- Success Judge is still primarily OCR/text based.
- AgentLoop currently executes a precomputed plan and lacks robust retry/recovery/replan.
- Real-task sample size is too small for a stable success-rate claim.
- PaddleOCR startup and repeated observation/OCR introduce significant latency.
- The project does not yet expose a polished end-user CLI/application entry point.

## Recommended next development priorities

1. Re-run the complete automated baseline and record the exact current result.
2. Establish a formal evaluation protocol: Auto Judge + Human Review + failure taxonomy.
3. Expand the real task suite beyond the initial five tasks.
4. Standardize module/task performance benchmarks and raw timing output.
5. Add retry/recovery/replan policies.
6. Add hybrid grounding for icons and non-text controls.

## Documentation

- [Project structure and architecture](docs/architecture/project_structure.md)
- [Code review and improvement backlog](docs/architecture/code_review.md)
- [GUI Agent v1 usage guide](docs/guides/week4_v1_usage.md)
- [Week 4 v1 report](docs/reports/week4_v1_report.md)
