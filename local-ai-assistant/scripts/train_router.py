#!/usr/bin/env python3
"""LoRA fine-tune Gemma 3 270M as a three-label causal classifier.

This script intentionally imports the ML stack lazily. Dataset creation and
runtime tests work on a normal Lura install; training additionally needs the
packages in requirements-training.txt and Hugging Face access to Gemma.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_ai_assistant.assistant_core import ROUTER_SYSTEM_PROMPT
from local_ai_assistant.router_dataset import LABELS


def _require_training_packages():
    try:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise SystemExit(
            "Training dependencies are missing. Install "
            "requirements-training.txt first."
        ) from error
    return (
        load_dataset,
        LoraConfig,
        get_peft_model,
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )


def _format_pair(tokenizer, text: str, label: str) -> tuple[str, str]:
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    assistant = {"role": "assistant", "content": label}
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full = tokenizer.apply_chat_template(
            [*messages, assistant], tokenize=False, add_generation_prompt=False
        )
        return prompt, full
    prompt = f"System: {ROUTER_SYSTEM_PROMPT}\nUser: {text}\nAssistant:"
    return prompt, f"{prompt} {label}"


def _tokenize_row(row, tokenizer, max_length: int) -> dict:
    prompt, full = _format_pair(tokenizer, row["text"], row["label"])
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=True)["input_ids"]
    if len(prompt_ids) >= max_length:
        raise ValueError(
            "Router prompt is too long for --max-length: "
            f"{len(prompt_ids)} prompt tokens >= {max_length}. "
            "Increase --max-length so the assistant label remains in the loss."
        )
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
    prompt_length = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_length + full_ids[prompt_length:]
    labels = labels[: len(full_ids)]
    if not any(label != -100 for label in labels):
        raise ValueError(
            "Router example has no trainable label tokens after truncation. "
            "Increase --max-length."
        )
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("training/router_data"))
    parser.add_argument("--model-id", default="google/gemma-3-270m-it")
    parser.add_argument("--output-dir", type=Path, default=Path("training/router_lora"))
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length; must leave room for the assistant label.",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--qlora", action="store_true", help="Use 4-bit loading (requires CUDA and bitsandbytes).")
    args = parser.parse_args()
    if args.max_length < 64 or args.batch_size < 1:
        parser.error("max-length must be >= 64 and batch-size must be positive")
    train_path = args.data_dir / "train.jsonl"
    validation_path = args.data_dir / "validation.jsonl"
    if not train_path.exists() or not validation_path.exists():
        raise SystemExit("Dataset is missing; run generate_router_dataset.py first.")

    (
        load_dataset,
        LoraConfig,
        get_peft_model,
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    ) = _require_training_packages()

    import torch

    if args.qlora and not torch.cuda.is_available():
        raise SystemExit("--qlora requires a CUDA runtime.")
    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(validation_path)},
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized = dataset.map(
        lambda row: _tokenize_row(row, tokenizer, args.max_length),
        remove_columns=dataset["train"].column_names,
    )
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {"torch_dtype": torch_dtype}
    if args.qlora:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    if args.qlora:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameter_names = inspect.signature(TrainingArguments.__init__).parameters
    training_kwargs = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )
    if "warmup_ratio" in parameter_names:
        training_kwargs["warmup_ratio"] = 0.05
    elif "warmup_steps" in parameter_names:
        steps_per_epoch = max(
            1,
            math.ceil(
                len(tokenized["train"])
                / (args.batch_size * args.gradient_accumulation)
            ),
        )
        training_kwargs["warmup_steps"] = max(
            1, round(steps_per_epoch * args.epochs * 0.05)
        )
    if "eval_strategy" in parameter_names:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in parameter_names:
        training_kwargs["evaluation_strategy"] = "epoch"
    for optional_name in (
        "save_strategy",
        "load_best_model_at_end",
        "metric_for_best_model",
        "greater_is_better",
        "report_to",
        "remove_unused_columns",
        "fp16",
        "bf16",
    ):
        if optional_name not in parameter_names:
            training_kwargs.pop(optional_name, None)
    train_args = TrainingArguments(**training_kwargs)
    trainer_kwargs = dict(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, label_pad_token_id=-100, padding=True, return_tensors="pt"
        ),
    )
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    (args.output_dir / "router_training_config.json").write_text(
        json.dumps(
            {
                "base_model": args.model_id,
                "method": "LoRA" if not args.qlora else "QLoRA",
                "labels": list(LABELS),
                "system_prompt_contract": "local_ai_assistant.assistant_core.ROUTER_SYSTEM_PROMPT",
                "max_length": args.max_length,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved adapter to {args.output_dir}")
    print("Run evaluate_router.py on the untouched test split before serving it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())