# GUI Agent

A desktop GUI agent project developed from scratch.

## Status

# Development Progress

## Week 1 — Research and Project Initialization

**Status: Completed**

### Technical Research

Investigated three representative GUI Agent approaches:

- UI-TARS: native GUI models and unified action representations.
- Claude Computer Use: computer tools and the Agent Loop.
- ScreenAgent: Planning, Action, and Reflection.

Studied the core concepts of LLMs, VLMs, Tool Calling, ReAct,
Visual Grounding, Planning, and Agent execution loops.

Selected a modular architecture for the initial implementation:

Perception → Grounding → Planning → Action → Feedback

### Environment and Project Setup

- Initialized the Git repository and Python project structure.
- Configured Windows, Conda, Python 3.11, and NVIDIA GPU support.
- Verified PyTorch CUDA availability.
- Implemented initial desktop screenshot experiments using MSS.
- Verified local Qwen3-VL screenshot understanding.

### Deliverables

- Technical research report.
- Initial project architecture and development environment.
- Basic screenshot and VLM inference experiments.

---

## Week 2 — Desktop Perception and Control

**Status: Completed**

### Screen Capture

- Implemented full-screen and region capture using MSS.
- Standardized screenshot output as BGR NumPy arrays.
- Added image-size and region-boundary validation.
- Verified coordinate alignment between MSS and PyAutoGUI.

### OCR

- Implemented a unified OCR interface and result structure.
- Integrated EasyOCR and PaddleOCR.
- Added text, confidence, and bounding-box extraction.
- Implemented OCR result visualization.
- Selected PaddleOCR GPU as the primary OCR backend.
- Retained EasyOCR as a fallback.

### UI Localization

- Implemented exact and fuzzy text matching.
- Added confidence filtering and ROI support.
- Added missing-target and ambiguous-target handling.
- Converted OCR bounding boxes into target center coordinates.

### Coordinate Mapping and Control

- Implemented image-to-desktop coordinate mapping.
- Supported region offsets and different scaling factors.
- Implemented mouse movement, clicking, double-clicking,
  dragging, scrolling, key presses, and hotkeys.
- Implemented Windows Unicode text input using SendInput.

### Integration

Verified the complete desktop interaction pipeline:

ScreenCapture
    ↓
PaddleOCR
    ↓
UILocator
    ↓
CoordinateMapper
    ↓
Controller

The end-to-end test successfully performed target recognition,
coordinate mapping, mouse clicking, and Chinese/English text input.

### Testing and Results

- 32/32 Week 2 acceptance checks passed.
- Expanded the stable baseline to 253 automated tests.
- Verified full desktop end-to-end interaction.
- Verified Torch and PaddleOCR GPU coexistence under the
  required initialization order.

OCR benchmark on the same test screenshot:

| OCR Backend | Detections | Inference Time |
|---|---:|---:|
| EasyOCR GPU | 208 | 4.170 s |
| PaddleOCR CPU | 202 | 32.560 s |
| PaddleOCR GPU | 202 | 1.919–1.971 s |

These measurements describe one test screenshot and hardware
configuration, not a general benchmark across datasets.

### Known Limitations

- Primary validation targets a single Windows display.
- OCR text bounding boxes do not always represent complete UI controls.
- Torch and PaddleOCR GPU coexistence depends on initialization order
  and a manually configured cuDNN environment.
- General-purpose VLM grounding and autonomous action feedback are
  not yet integrated into the stable execution pipeline.

### Deliverables

- Desktop perception and control modules.
- OCR benchmarking scripts.
- End-to-end integration test.
- Week 2 test report.
- Stable environment snapshots.