# Development scripts

`scripts/` contains manually executed developer workflows. These files are intentionally separated from `tests/` because they may manipulate the real desktop, call external APIs, load large models, or process large datasets.

## `benchmarks/`

Performance-oriented runs and benchmark input preparation.

- `save_ocr_benchmark.py` — capture a benchmark screenshot into `artifacts/ocr/`.
- `benchmark_ocr.py` — compare OCR inference on a saved image.
- `benchmark_local_vlm.py` — measure local VLM load/inference timing.

## `datasets/`

Dataset export, integrity checking, splitting, preprocessing, and auditing.

- ScreenAgent: export, audit, integrity check, split, planner dataset build.
- Mind2Web: batch export.
- WebArena: audit and real-data parser check.

Generated/large datasets live under `data/` and are ignored by Git.

## `diagnostics/`

Environment-dependent checks that should not run as normal pytest tests.

- `test_gpu_coexistence.py` — Torch → PaddleOCR GPU → Torch validation.
- `test_local_vlm.py` — real local VLM inference.
- `test_api_real.py` — real remote multimodal API request.

## `e2e/`

Real desktop workflows.

- `e2e_target_app.py` — small target application for desktop interaction tests.
- `test_e2e.py` — Week 2 perception/localization/control E2E.
- `test_agent_chain_e2e.py` — planning-chain E2E.
- `test_planner_e2e.py` — real planner E2E.
- `test_week4_real_e2e.py` — real Week 4 agent-loop E2E.
- `week4_task_suite.py` — five-task Week 4 baseline suite.

## Safety

Run real desktop scripts only when you are ready for the agent to move the mouse/type/open applications. Keep API keys in environment variables. Do not print or commit sensitive full-screen OCR output.
