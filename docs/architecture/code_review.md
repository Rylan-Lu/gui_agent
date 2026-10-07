# Code review and improvement backlog

This review focuses on clarity, reliability, maintainability, and the path from the Week 4 engineering baseline toward a stronger production-style GUI agent.

## Changes applied in this cleanup

### Repository organization

- Removed generated IDE/cache/package metadata from the cleaned project copy.
- Split the previously flat `scripts/` directory into `benchmarks/`, `datasets/`, `diagnostics/`, and `e2e/`.
- Split documentation into `architecture/`, `guides/`, and `reports/`.
- Centralized generated OCR benchmark images under `artifacts/ocr/`.
- Moved Week 4 generated E2E files to `artifacts/e2e/week4/`.
- Removed import-time file generation from the Week 4 task suite.

### Correctness and code clarity

- Tightened `Observation` and runtime observation type annotations.
- Added explicit OCR-result typing to the Grounding bridge.
- Routed local VLM loading through the shared Torch-first GPU initialization helper.
- Made EasyOCR's GPU usage configurable so CPU-only diagnostics are possible.
- Fixed `TextAbsentJudge` so `ocr_result=None` is not incorrectly treated as proof that text is absent.
- Added compact-normalization support to `TextAbsentJudge`, matching the robustness already present in the positive text judge.
- Added regression tests for the new absence-judge behavior and EasyOCR CPU configuration.

## High-priority remaining issues

### P0 — establish the real current baseline

The historical full suite reached 427 passed before later Success Judge changes. Run the complete suite on the validated Windows environment and record the exact current count before further feature development.

### P0 — formalize task success evaluation

The current Success Judge is predominantly OCR text based. A screen can change without the requested task being correct, and OCR can miss visual state. Formal evaluation should record both automated judge output and human review until the automatic judge covers more task types.

### P1 — retry / recovery / replan

`AgentLoop` currently executes a precomputed sequence. An execution exception ends the task; a semantically wrong but non-throwing action can continue through the plan. Add bounded retry/re-observe/replan policies rather than only increasing `max_steps`.

### P1 — hybrid grounding

`GroundingBridge` currently resolves unresolved click/double-click actions through OCR text. It does not provide robust grounding for icons, unlabeled visual controls, canvas elements, or accessibility-only controls.

Recommended future order for Windows:

1. UI Automation/accessibility tree when available.
2. OCR text grounding for visible labels.
3. VLM/vision grounding as the expensive fallback.

This ordering is also more suitable for lower-spec SME devices than running a VLM on every action.

### P1 — performance instrumentation

Current timing data comes from separate scripts and small runs. Introduce one benchmark result schema containing at least:

- cold/warm flag,
- run index,
- elapsed time,
- module/task name,
- input identifier,
- CPU/GPU mode,
- optional RAM/VRAM measurements.

Save raw rows and compute Mean/P50/P95/Std afterward. Avoid mixing model-load time with inference time.

### P1 — observation cost

`ActionTransaction` normally captures both a before and after screenshot, and unresolved text clicks additionally OCR the before state. Final judging may capture/OCR again. This is easy to understand but can dominate latency.

Optimization should be measured before changing behavior. Candidate improvements include observation reuse, ROI OCR, lazy OCR, and action-specific feedback policies.

## Medium-priority issues

### P2 — product entry point

The repository currently demonstrates end-to-end behavior through scripts. A real application entry point (`python -m gui_agent` or a console script) and a composition/bootstrap layer would make the system easier to run and configure without duplicating construction code in E2E scripts.

### P2 — configuration

Model IDs, OCR device selection, confidence thresholds, timeouts, and settle times are currently spread across constructors/scripts. Centralized typed configuration should be introduced when multiple deployment profiles are needed.

### P2 — API reliability

`APIModelClient` deliberately keeps a small stdlib HTTP surface, but production use should eventually add categorized HTTP/network errors, bounded retry/backoff for transient failures, request IDs/telemetry, and explicit rate-limit handling.

### P2 — local VLM deployment flexibility

The local VLM path is CUDA-only and assumes model files are already available locally. For lower-spec machines, support a documented remote-model mode and evaluate quantized/smaller local models before adding complex deployment machinery.

### P2 — package boundaries

The `agent/` package contains planning, execution bridge, runtime, and evaluation code. It is still manageable at the current size. Split only when growth makes the boundary useful; do not move files solely for aesthetic reasons.

### P2 — duplicated geometry aliases

`Point`, `Region`, and `ImageSize` aliases occur in several modules. A shared geometry/types module could reduce drift later, but this is not currently worth a large refactor because the definitions are simple and stable.

## Performance principles for SME deployment

1. Prefer cheap structured signals before expensive vision inference.
2. Avoid full-screen OCR/VLM when a window/ROI is known.
3. Cache observations within the same decision step when correctness permits.
4. Separate cold-start and warm-path measurements.
5. Keep CPU-only fallbacks for basic perception/control diagnostics.
6. Treat model loading, OCR startup, and repeated screenshots as first-class latency sources.
7. Optimize only after module-level profiling identifies the bottleneck.

## Validation performed during this review

The uploaded project snapshot originally collected **429 pytest cases** in the review environment. After the cleanup and the five added regression cases, the organized copy collected **434 passed**.

Because the review container is not the validated Windows workstation, lightweight shims were used for unavailable `mss`/`langchain_core` imports and Xvfb was used for GUI-dependent imports. This result verifies Python-level regressions in the reviewed copy; it does **not** replace the required Windows/GPU regression run, `pip check`, GPU coexistence check, or real desktop E2E.

`python -m compileall` also completed successfully for `src/`, `scripts/`, and `tests/`.
