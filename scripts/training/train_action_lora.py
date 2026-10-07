from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

ACTION_SFT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "screenagent"
    / "action_sft"
)

DEFAULT_TRAIN_DATASET = (
    ACTION_SFT_ROOT
    / "train.jsonl"
)

DEFAULT_DEV_DATASET = (
    ACTION_SFT_ROOT
    / "dev.jsonl"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "week5"
    / "action_lora_full"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a Qwen3-VL "
            "LoRA adapter for step-level "
            "GUI action prediction."
        )
    )

    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=DEFAULT_TRAIN_DATASET,
    )

    parser.add_argument(
        "--dev-dataset",
        type=Path,
        default=DEFAULT_DEV_DATASET,
    )

    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--model-id",
        default=(
            "Qwen/Qwen3-VL-2B-Instruct"
        ),
    )

    parser.add_argument(
        "--prompt-version",
        choices=(
            "base",
            "optimized",
            "compact",
        ),
        default="compact",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help=(
            "Optional stratified dev-set "
            "evaluation limit. "
            "Default evaluates all dev samples."
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Optional maximum optimizer "
            "steps across all epochs."
        ),
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    if args.epochs <= 0:
        raise ValueError(
            "epochs must be positive"
        )

    if (
        args.gradient_accumulation_steps
        <= 0
    ):
        raise ValueError(
            "gradient accumulation steps "
            "must be positive"
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "learning rate must be positive"
        )

    if args.max_grad_norm <= 0:
        raise ValueError(
            "max_grad_norm must be positive"
        )

    if args.max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must be positive"
        )

    if args.lora_r <= 0:
        raise ValueError(
            "LoRA rank must be positive"
        )

    if args.lora_alpha <= 0:
        raise ValueError(
            "LoRA alpha must be positive"
        )

    if not (
        0 <= args.lora_dropout < 1
    ):
        raise ValueError(
            "LoRA dropout must be in [0, 1)"
        )

    if (
        args.max_train_samples
        is not None
        and args.max_train_samples <= 0
    ):
        raise ValueError(
            "max_train_samples must "
            "be positive"
        )

    if (
        args.eval_limit is not None
        and args.eval_limit <= 0
    ):
        raise ValueError(
            "eval_limit must be positive"
        )

    if (
        args.max_steps is not None
        and args.max_steps <= 0
    ):
        raise ValueError(
            "max_steps must be positive"
        )

    if not args.image_root.is_dir():
        raise FileNotFoundError(
            args.image_root
        )


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    samples: list[
        dict[str, Any]
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                sample = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at "
                    f"{path}:{line_number}"
                ) from exc

            if not isinstance(
                sample,
                dict,
            ):
                raise ValueError(
                    "sample must be "
                    f"an object at "
                    f"{path}:{line_number}"
                )

            samples.append(
                sample
            )

    if not samples:
        raise ValueError(
            f"dataset is empty: {path}"
        )

    return samples


def choose_training_samples(
    samples: list[
        dict[str, Any]
    ],
    *,
    limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    selected = list(
        samples
    )

    if limit is None:
        return selected

    rng = random.Random(
        seed
    )

    rng.shuffle(
        selected
    )

    return selected[:limit]


def build_target_text(
    sample: dict[str, Any],
) -> str:
    target = sample.get(
        "target"
    )

    if not isinstance(
        target,
        dict,
    ):
        raise ValueError(
            "sample target "
            "must be an object"
        )

    return json.dumps(
        target,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def common_prefix_length(
    left,
    right,
) -> int:
    if (
        left.ndim != 2
        or right.ndim != 2
    ):
        raise ValueError(
            "token tensors "
            "must be rank 2"
        )

    if (
        left.shape[0] != 1
        or right.shape[0] != 1
    ):
        raise ValueError(
            "only batch size 1 "
            "is supported"
        )

    limit = min(
        left.shape[1],
        right.shape[1],
    )

    if limit == 0:
        return 0

    equal = (
        left[0, :limit]
        == right[0, :limit]
    )

    mismatch = (
        ~equal
    ).nonzero(
        as_tuple=False
    )

    if mismatch.numel() == 0:
        return limit

    return int(
        mismatch[
            0,
            0,
        ].item()
    )


def prepare_training_inputs(
    *,
    processor,
    sample: dict[str, Any],
    image_path: Path,
    system_prompt: str,
    user_prompt: str,
):
    import torch

    if not image_path.is_file():
        raise FileNotFoundError(
            image_path
        )

    target_text = (
        build_target_text(
            sample
        )
    )

    with Image.open(
        image_path
    ) as raw_image:
        image = raw_image.convert(
            "RGB"
        )

    user_content = [
        {
            "type": "image",
            "image": image,
        },
        {
            "type": "text",
            "text": user_prompt,
        },
    ]

    prompt_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    full_messages = [
        *prompt_messages,
        {
            "role": "assistant",
            "content": target_text,
        },
    ]

    prompt_inputs = (
        processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    )

    full_inputs = (
        processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
    )

    image.close()

    prompt_inputs.pop(
        "token_type_ids",
        None,
    )

    full_inputs.pop(
        "token_type_ids",
        None,
    )

    prompt_ids = (
        prompt_inputs[
            "input_ids"
        ]
    )

    full_ids = (
        full_inputs[
            "input_ids"
        ]
    )

    prefix_length = (
        common_prefix_length(
            prompt_ids,
            full_ids,
        )
    )

    if (
        prefix_length
        < prompt_ids.shape[1]
    ):
        raise RuntimeError(
            "prompt/full chat templates "
            "do not share the expected "
            "token prefix: "
            f"common={prefix_length}, "
            f"prompt="
            f"{prompt_ids.shape[1]}, "
            f"full="
            f"{full_ids.shape[1]}"
        )

    if (
        full_ids.shape[1]
        <= prefix_length
    ):
        raise RuntimeError(
            "training example has "
            "no assistant target tokens"
        )

    labels = (
        full_ids.clone()
    )

    labels[
        :,
        :prefix_length,
    ] = -100

    if not torch.any(
        labels != -100
    ):
        raise RuntimeError(
            "all training labels "
            "were masked"
        )

    full_inputs[
        "labels"
    ] = labels

    target_tokens = int(
        (
            labels != -100
        ).sum().item()
    )

    return (
        full_inputs,
        target_text,
        prefix_length,
        target_tokens,
    )


def generate_prediction(
    *,
    model,
    processor,
    sample: dict[str, Any],
    image_path: Path,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
) -> tuple[str, float]:
    import torch

    if not image_path.is_file():
        raise FileNotFoundError(
            image_path
        )

    with Image.open(
        image_path
    ) as raw_image:
        image = raw_image.convert(
            "RGB"
        )

    messages = [
        {
            "role": "system",
            "content": (
                system_prompt
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": user_prompt,
                },
            ],
        },
    ]

    inputs = (
        processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    )

    image.close()

    inputs.pop(
        "token_type_ids",
        None,
    )

    inputs = inputs.to(
        model.device
    )

    input_length = (
        inputs[
            "input_ids"
        ].shape[1]
    )

    torch.cuda.synchronize()

    start = (
        time.perf_counter()
    )

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=(
                max_new_tokens
            ),
            do_sample=False,
            use_cache=False,
        )

    torch.cuda.synchronize()

    latency = (
        time.perf_counter()
        - start
    )

    generated_ids = generated[
        :,
        input_length:,
    ]

    text = (
        processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    )

    del inputs
    del generated
    del generated_ids

    return (
        text.strip(),
        latency,
    )


def evaluate_dev(
    *,
    model,
    processor,
    samples: list[
        dict[str, Any]
    ],
    image_root: Path,
    output_dir: Path,
    system_prompt: str,
    prompt_version: str,
    max_new_tokens: int,
    model_id: str,
) -> dict[str, Any]:
    import torch

    from gui_agent.evaluation.action_eval import (
        build_action_policy_prompt,
        parse_json_object,
        score_prediction,
        summarize_results,
        validate_action_prediction,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        output_dir
        / "results.jsonl"
    )

    model.eval()

    results: list[
        dict[str, Any]
    ] = []

    with results_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for index, sample in enumerate(
            samples,
            start=1,
        ):
            image_path = (
                image_root
                / sample["screenshot"]
            )

            user_prompt = (
                build_action_policy_prompt(
                    sample
                )
            )

            raw_output, latency = (
                generate_prediction(
                    model=model,
                    processor=processor,
                    sample=sample,
                    image_path=image_path,
                    system_prompt=(
                        system_prompt
                    ),
                    user_prompt=(
                        user_prompt
                    ),
                    max_new_tokens=(
                        max_new_tokens
                    ),
                )
            )

            json_parsed = False
            schema_valid = False
            prediction = None
            parse_error = None

            try:
                parsed = (
                    parse_json_object(
                        raw_output
                    )
                )

                json_parsed = True

                prediction = (
                    validate_action_prediction(
                        parsed,
                        screen_size=(
                            sample[
                                "screen_size"
                            ]
                        ),
                    )
                )

                schema_valid = True

            except (
                ValueError,
                TypeError,
                json.JSONDecodeError,
            ) as exc:
                parse_error = str(
                    exc
                )

            score = score_prediction(
                sample["target"],
                prediction,
            )

            result = {
                "sample_id": (
                    sample[
                        "sample_id"
                    ]
                ),
                "prompt_version": (
                    prompt_version
                ),
                "target": (
                    sample[
                        "target"
                    ]
                ),
                "prediction": (
                    prediction
                ),
                "raw_output": (
                    raw_output
                ),
                "json_parsed": (
                    json_parsed
                ),
                "schema_valid": (
                    schema_valid
                ),
                "parse_error": (
                    parse_error
                ),
                "score": (
                    score
                ),
                "latency_seconds": (
                    latency
                ),
                "model_name": (
                    model_id
                ),
            }

            results.append(
                result
            )

            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            file.flush()

            print(
                f"  dev "
                f"[{index}/{len(samples)}] "
                f"target="
                f"{sample['target']['action_type']} "
                f"json={json_parsed} "
                f"schema={schema_valid} "
                f"type_ok="
                f"{score['action_type_correct']}"
            )

    metrics = (
        summarize_results(
            results
        )
    )

    summary = {
        "samples": (
            len(samples)
        ),
        "prompt_version": (
            prompt_version
        ),
        "metrics": (
            metrics
        ),
    }

    summary_path = (
        output_dir
        / "summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    model.train()

    torch.cuda.empty_cache()

    return summary


def dev_metric_key(
    dev_summary: dict[
        str,
        Any,
    ],
) -> tuple[
    float,
    float,
    float,
    float,
]:
    metrics = (
        dev_summary[
            "metrics"
        ]
    )

    return (
        float(
            metrics[
                "success_at_50px"
            ]
        ),
        float(
            metrics[
                "exact_action_accuracy"
            ]
        ),
        float(
            metrics[
                "action_type_accuracy"
            ]
        ),
        float(
            metrics[
                "schema_valid_rate"
            ]
        ),
    )


def save_adapter(
    *,
    model,
    path: Path,
) -> None:
    if path.exists():
        shutil.rmtree(
            path
        )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        path
    )


def write_json(
    path: Path,
    data: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")


def main() -> None:
    args = parse_args()

    validate_args(
        args
    )

    import torch

    from peft import (
        LoraConfig,
        get_peft_model,
    )

    from transformers import (
        AutoProcessor,
        Qwen3VLForConditionalGeneration,
    )

    from gui_agent.evaluation.action_eval import (
        ACTION_POLICY_PROMPTS,
        build_action_policy_prompt,
        select_samples,
    )

    from gui_agent.runtime.gpu_runtime import (
        initialize_torch_gpu_runtime,
    )

    initialize_torch_gpu_runtime()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required "
            "for LoRA training"
        )

    if not (
        torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "BF16 is required "
            "by this configuration"
        )

    random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    torch.cuda.manual_seed_all(
        args.seed
    )

    train_samples = (
        load_jsonl(
            args.train_dataset
        )
    )

    train_samples = (
        choose_training_samples(
            train_samples,
            limit=(
                args.max_train_samples
            ),
            seed=args.seed,
        )
    )

    dev_samples = (
        load_jsonl(
            args.dev_dataset
        )
    )

    dev_samples = (
        select_samples(
            dev_samples,
            limit=args.eval_limit,
            seed=args.seed,
        )
    )

    system_prompt = (
        ACTION_POLICY_PROMPTS[
            args.prompt_version
        ]
    )

    output_dir = (
        args.output.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logs_dir = (
        output_dir
        / "logs"
    )

    checkpoints_dir = (
        output_dir
        / "checkpoints"
    )

    eval_dir = (
        output_dir
        / "dev_eval"
    )

    best_adapter_dir = (
        output_dir
        / "best_adapter"
    )

    final_adapter_dir = (
        output_dir
        / "final_adapter"
    )

    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoints_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    eval_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_config = {
        "model_id": (
            args.model_id
        ),
        "train_dataset": str(
            args.train_dataset.resolve()
        ),
        "dev_dataset": str(
            args.dev_dataset.resolve()
        ),
        "image_root": str(
            args.image_root.resolve()
        ),
        "prompt_version": (
            args.prompt_version
        ),
        "epochs": (
            args.epochs
        ),
        "train_samples": (
            len(train_samples)
        ),
        "dev_samples": (
            len(dev_samples)
        ),
        "gradient_accumulation_steps": (
            args.gradient_accumulation_steps
        ),
        "learning_rate": (
            args.learning_rate
        ),
        "weight_decay": (
            args.weight_decay
        ),
        "max_grad_norm": (
            args.max_grad_norm
        ),
        "max_new_tokens": (
            args.max_new_tokens
        ),
        "seed": (
            args.seed
        ),
        "dtype": (
            "bfloat16"
        ),
        "lora": {
            "r": (
                args.lora_r
            ),
            "alpha": (
                args.lora_alpha
            ),
            "dropout": (
                args.lora_dropout
            ),
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
        },
    }

    write_json(
        output_dir
        / "run_config.json",
        run_config,
    )

    print(
        "===== FORMAL ACTION LORA ====="
    )

    print(
        "Train samples:",
        len(train_samples),
    )

    print(
        "Dev samples:",
        len(dev_samples),
    )

    print(
        "Epochs:",
        args.epochs,
    )

    print(
        "Prompt:",
        args.prompt_version,
    )

    print(
        "Gradient accumulation:",
        args.gradient_accumulation_steps,
    )

    print(
        "Learning rate:",
        args.learning_rate,
    )

    print(
        "\nLoading processor..."
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            args.model_id,
            local_files_only=True,
        )
    )

    print(
        "Loading base model..."
    )

    model = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(
            args.model_id,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
            local_files_only=True,
        )
    )

    # Disable KV cache explicitly for
    # gradient-checkpointed training.
    model.config.use_cache = False

    base_inner = getattr(
        model,
        "model",
        None,
    )

    language_model = getattr(
        base_inner,
        "language_model",
        None,
    )

    if (
        language_model is not None
        and hasattr(
            language_model,
            "config",
        )
    ):
        language_model.config.use_cache = (
            False
        )

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        }
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=(
            args.lora_alpha
        ),
        lora_dropout=(
            args.lora_dropout
        ),
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.train()

    model.print_trainable_parameters()

    trainable_parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    trainable_count = sum(
        parameter.numel()
        for parameter
        in trainable_parameters
    )

    total_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=(
            args.weight_decay
        ),
    )

    global_step = 0
    global_micro_step = 0

    training_start = (
        time.perf_counter()
    )

    torch.cuda.reset_peak_memory_stats()

    step_log_path = (
        logs_dir
        / "train_steps.jsonl"
    )

    epoch_records: list[
        dict[str, Any]
    ] = []

    best_epoch = None
    best_key = None
    stop_training = False

    with step_log_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as step_log:
        for epoch_index in range(
            1,
            args.epochs + 1,
        ):
            epoch_start = (
                time.perf_counter()
            )

            epoch_samples = list(
                train_samples
            )

            epoch_rng = (
                random.Random(
                    args.seed
                    + epoch_index
                    - 1
                )
            )

            epoch_rng.shuffle(
                epoch_samples
            )

            epoch_losses: list[
                float
            ] = []

            epoch_target_tokens = 0

            print(
                f"\n===== EPOCH "
                f"{epoch_index}/"
                f"{args.epochs} ====="
            )

            for group_start in range(
                0,
                len(epoch_samples),
                args.gradient_accumulation_steps,
            ):
                group = epoch_samples[
                    group_start:
                    group_start
                    + args.gradient_accumulation_steps
                ]

                if not group:
                    continue

                optimizer.zero_grad(
                    set_to_none=True
                )

                group_losses: list[
                    float
                ] = []

                for sample in group:
                    image_path = (
                        args.image_root
                        / sample[
                            "screenshot"
                        ]
                    )

                    user_prompt = (
                        build_action_policy_prompt(
                            sample
                        )
                    )

                    (
                        inputs,
                        target_text,
                        prompt_tokens,
                        target_tokens,
                    ) = (
                        prepare_training_inputs(
                            processor=processor,
                            sample=sample,
                            image_path=(
                                image_path
                            ),
                            system_prompt=(
                                system_prompt
                            ),
                            user_prompt=(
                                user_prompt
                            ),
                        )
                    )

                    inputs = inputs.to(
                        model.device
                    )

                    outputs = model(
                        **inputs
                    )

                    loss = (
                        outputs.loss
                    )

                    if (
                        loss is None
                        or not torch.isfinite(
                            loss
                        )
                    ):
                        raise RuntimeError(
                            "non-finite loss at "
                            f"{sample['sample_id']}: "
                            f"{loss}"
                        )

                    raw_loss = float(
                        loss.detach().item()
                    )

                    scaled_loss = (
                        loss
                        / len(group)
                    )

                    scaled_loss.backward()

                    group_losses.append(
                        raw_loss
                    )

                    epoch_losses.append(
                        raw_loss
                    )

                    epoch_target_tokens += (
                        target_tokens
                    )

                    global_micro_step += 1

                    del inputs
                    del outputs
                    del loss
                    del scaled_loss

                grad_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(
                        trainable_parameters,
                        args.max_grad_norm,
                    )
                )

                optimizer.step()

                global_step += 1

                step_loss = mean(
                    group_losses
                )

                allocated = (
                    torch.cuda
                    .memory_allocated()
                    / 1024**3
                )

                reserved = (
                    torch.cuda
                    .memory_reserved()
                    / 1024**3
                )

                step_record = {
                    "epoch": (
                        epoch_index
                    ),
                    "global_step": (
                        global_step
                    ),
                    "global_micro_step": (
                        global_micro_step
                    ),
                    "group_size": (
                        len(group)
                    ),
                    "loss": (
                        step_loss
                    ),
                    "grad_norm": (
                        float(
                            grad_norm
                        )
                    ),
                    "gpu_allocated_gib": (
                        allocated
                    ),
                    "gpu_reserved_gib": (
                        reserved
                    ),
                }

                step_log.write(
                    json.dumps(
                        step_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                step_log.flush()

                print(
                    f"epoch={epoch_index} "
                    f"step={global_step} "
                    f"micro="
                    f"{global_micro_step} "
                    f"loss="
                    f"{step_loss:.6f} "
                    f"grad_norm="
                    f"{float(grad_norm):.4f} "
                    f"allocated="
                    f"{allocated:.2f}GiB "
                    f"reserved="
                    f"{reserved:.2f}GiB"
                )

                if (
                    args.max_steps
                    is not None
                    and global_step
                    >= args.max_steps
                ):
                    stop_training = True
                    break

            epoch_train_seconds = (
                time.perf_counter()
                - epoch_start
            )

            epoch_train_summary = {
                "epoch": (
                    epoch_index
                ),
                "optimizer_steps_total": (
                    global_step
                ),
                "micro_steps_total": (
                    global_micro_step
                ),
                "train_samples_seen": (
                    len(epoch_losses)
                ),
                "mean_train_loss": (
                    mean(epoch_losses)
                    if epoch_losses
                    else None
                ),
                "min_train_loss": (
                    min(epoch_losses)
                    if epoch_losses
                    else None
                ),
                "max_train_loss": (
                    max(epoch_losses)
                    if epoch_losses
                    else None
                ),
                "target_tokens": (
                    epoch_target_tokens
                ),
                "train_seconds": (
                    epoch_train_seconds
                ),
            }

            checkpoint_adapter = (
                checkpoints_dir
                / (
                    f"epoch_"
                    f"{epoch_index:03d}"
                )
                / "adapter"
            )

            print(
                "\nSaving epoch "
                f"{epoch_index} checkpoint..."
            )

            save_adapter(
                model=model,
                path=(
                    checkpoint_adapter
                ),
            )

            # Release unused cached CUDA
            # blocks before generation.
            torch.cuda.empty_cache()

            print(
                "\nRunning dev evaluation..."
            )

            dev_summary = (
                evaluate_dev(
                    model=model,
                    processor=processor,
                    samples=dev_samples,
                    image_root=(
                        args.image_root
                    ),
                    output_dir=(
                        eval_dir
                        / (
                            f"epoch_"
                            f"{epoch_index:03d}"
                        )
                    ),
                    system_prompt=(
                        system_prompt
                    ),
                    prompt_version=(
                        args.prompt_version
                    ),
                    max_new_tokens=(
                        args.max_new_tokens
                    ),
                    model_id=(
                        args.model_id
                    ),
                )
            )

            current_key = (
                dev_metric_key(
                    dev_summary
                )
            )

            is_best = (
                best_key is None
                or current_key
                > best_key
            )

            if is_best:
                best_key = (
                    current_key
                )

                best_epoch = (
                    epoch_index
                )

                print(
                    "\nNew best adapter "
                    f"at epoch "
                    f"{epoch_index}"
                )

                save_adapter(
                    model=model,
                    path=(
                        best_adapter_dir
                    ),
                )

            epoch_record = {
                **epoch_train_summary,
                "checkpoint_adapter": str(
                    checkpoint_adapter
                ),
                "dev": (
                    dev_summary[
                        "metrics"
                    ]
                ),
                "best": (
                    is_best
                ),
            }

            epoch_records.append(
                epoch_record
            )

            write_json(
                logs_dir
                / (
                    f"epoch_"
                    f"{epoch_index:03d}"
                    f"_summary.json"
                ),
                epoch_record,
            )

            metrics = (
                dev_summary[
                    "metrics"
                ]
            )

            print(
                "\nEPOCH SUMMARY"
            )

            print(
                "mean_train_loss:",
                epoch_train_summary[
                    "mean_train_loss"
                ],
            )

            print(
                "dev_json_parse:",
                metrics[
                    "json_parse_rate"
                ],
            )

            print(
                "dev_schema_valid:",
                metrics[
                    "schema_valid_rate"
                ],
            )

            print(
                "dev_action_type:",
                metrics[
                    "action_type_accuracy"
                ],
            )

            print(
                "dev_exact_action:",
                metrics[
                    "exact_action_accuracy"
                ],
            )

            print(
                "dev_success@50:",
                metrics[
                    "success_at_50px"
                ],
            )

            if stop_training:
                break

    print(
        "\nSaving final adapter..."
    )

    save_adapter(
        model=model,
        path=final_adapter_dir,
    )

    elapsed = (
        time.perf_counter()
        - training_start
    )

    peak_allocated = (
        torch.cuda
        .max_memory_allocated()
        / 1024**3
    )

    peak_reserved = (
        torch.cuda
        .max_memory_reserved()
        / 1024**3
    )

    final_summary = {
        **run_config,
        "optimizer_steps": (
            global_step
        ),
        "micro_steps": (
            global_micro_step
        ),
        "trainable_parameters": (
            trainable_count
        ),
        "total_parameters": (
            total_count
        ),
        "trainable_percentage": (
            100.0
            * trainable_count
            / total_count
        ),
        "elapsed_seconds": (
            elapsed
        ),
        "peak_gpu_memory_gib": {
            "allocated": (
                peak_allocated
            ),
            "reserved": (
                peak_reserved
            ),
        },
        "best_epoch": (
            best_epoch
        ),
        "best_metric_key": (
            list(best_key)
            if best_key is not None
            else None
        ),
        "best_adapter": str(
            best_adapter_dir
        ),
        "final_adapter": str(
            final_adapter_dir
        ),
        "epochs": (
            epoch_records
        ),
    }

    write_json(
        output_dir
        / "training_summary.json",
        final_summary,
    )

    print(
        "\nFORMAL ACTION LORA "
        "TRAINING: PASS"
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best adapter:",
        best_adapter_dir,
    )

    print(
        "Final adapter:",
        final_adapter_dir,
    )

    print(
        "Peak allocated:",
        f"{peak_allocated:.2f} GiB",
    )

    print(
        "Peak reserved:",
        f"{peak_reserved:.2f} GiB",
    )

    print(
        "Summary:",
        output_dir
        / "training_summary.json",
    )


if __name__ == "__main__":
    main()