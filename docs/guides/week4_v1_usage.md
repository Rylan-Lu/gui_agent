# GUI Agent v1.0 usage guide

## Validated environment

- Windows 11
- Conda environment: `gui-agent`
- Python 3.11.15
- NVIDIA RTX 5070 Ti Laptop GPU

The validated GPU setup is import-order sensitive: initialize Torch CUDA/cuDNN before loading Paddle/PaddleOCR. Do not casually reinstall or upgrade the working Torch/Paddle/OpenCV stack before capturing the current environment state.

## Automated regression

From the repository root:

```powershell
conda activate gui-agent
python -m pytest -q
python -m pip check
```

Use the terminal output as the authoritative current baseline. The historical Week 4 checkpoint reached 427 passed before later Success Judge changes.

## OCR benchmark

Capture the benchmark image:

```powershell
python scripts/benchmarks/save_ocr_benchmark.py
```

Run PaddleOCR:

```powershell
python scripts/benchmarks/benchmark_ocr.py --engine paddle
```

Run EasyOCR:

```powershell
python scripts/benchmarks/benchmark_ocr.py --engine easyocr
```

The generated image is stored under `artifacts/ocr/` and is ignored by Git.

## GPU coexistence diagnostic

```powershell
python scripts/diagnostics/test_gpu_coexistence.py
```

The expected execution order is:

```text
Torch GPU operation
→ PaddleOCR GPU inference
→ Torch GPU operation
```

## Local VLM / API diagnostics

```powershell
python scripts/diagnostics/test_local_vlm.py
python scripts/diagnostics/test_api_real.py --endpoint <HTTPS_CHAT_COMPLETIONS_ENDPOINT> --model <MODEL_ID>
```

For remote API use, set the key through the environment:

```powershell
$env:GUI_AGENT_API_KEY = "..."
```

Never place API keys in code, logs, Markdown reports, or Git.

## Real desktop E2E

These commands can manipulate the real desktop. Save important work first and keep PyAutoGUI failsafe behavior enabled.

Week 2-style desktop E2E:

```powershell
python scripts/e2e/test_e2e.py
```

Week 4 real AgentLoop E2E:

```powershell
python scripts/e2e/test_week4_real_e2e.py
```

Five-task baseline suite:

```powershell
python scripts/e2e/week4_task_suite.py
```

Generated task files are written under `artifacts/e2e/week4/`.

## Dataset tooling

Dataset preprocessing and auditing commands are grouped under:

```text
scripts/datasets/
```

Large/raw/generated datasets remain under `data/` and are intentionally ignored by Git.

## Known limitations

- OCR grounding primarily supports visible text elements.
- Success Judge is primarily OCR/text based.
- Retry/recovery/replan is not yet implemented as a robust policy.
- Real-task baseline size is small.
- Full-screen OCR/model inference remains an important performance cost.
