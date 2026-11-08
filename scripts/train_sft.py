import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from scripts.generate import MODEL_ID
from scripts.verifier import verify_answer


ROOT = Path(__file__).resolve().parents[1]


def read_rows(name):
    path = ROOT / "data" / f"{name}.jsonl"

    with path.open(encoding="utf-8") as file:
        rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not rows:
        raise ValueError(f"データが空です：{path}")

    return rows


def prepare_dataset(rows, tokenizer):
    examples = []

    for row in rows:
        grade = verify_answer(
            answer=row["completion"],
            numbers=row["numbers"],
            target=row["target"],
        )

        if not grade["correct"]:
            raise ValueError(
                f"不正な正解例：{row['id']} / {grade['reason']}"
            )

        # 評価時と同じ会話書式を使う
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        completion = row["completion"] + tokenizer.eos_token

        ids = tokenizer(
            prompt + completion,
            add_special_tokens=False,
        )["input_ids"]

        if len(ids) > 512:
            raise ValueError(
                f"512トークンを超える例：{row['id']}"
            )

        examples.append({
            "prompt": prompt,
            "completion": completion,
        })

    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--name", default="sft-pilot")
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--stepsは1以上にしてください")

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDAが利用できません")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPSが利用できません")

    output_dir = ROOT / "checkpoints" / args.name

    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} が既にあります。--nameを変更してください"
        )

    # まずFP32で統一。メモリ使用量は推論時より増える
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.config.use_cache = False

    train_dataset = prepare_dataset(
        read_rows("sft_train"), tokenizer
    )
    valid_dataset = prepare_dataset(
        read_rows("valid"), tokenizer
    )

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    config = SFTConfig(
        output_dir=str(output_dir),

        max_steps=args.steps,
        learning_rate=1e-4,
        lr_scheduler_type="constant",
        warmup_steps=0,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,

        max_length=512,
        packing=False,
        completion_only_loss=True,

        bf16=False,
        fp16=False,
        use_cpu=(device == "cpu"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        optim="adamw_torch",
        max_grad_norm=1.0,

        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",

        seed=42,
        report_to="none",
        push_to_hub=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=lora,
    )

    trainer.model.print_trainable_parameters()

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    started = time.perf_counter()
    trainer.train()

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - started

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    report = {
        "base_model": MODEL_ID,
        "device": device,
        "steps": trainer.state.global_step,
        "elapsed_seconds": elapsed,
        "seconds_per_update": (
            elapsed / trainer.state.global_step
        ),
        "adapter_path": str(output_dir),
        "history": trainer.state.log_history,
    }

    (results_dir / f"{args.name}-training.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n学習時間：{elapsed:.1f}秒")
    print(
        "1更新あたり："
        f"{report['seconds_per_update']:.1f}秒"
    )
    print(f"LoRA保存先：{output_dir}")


if __name__ == "__main__":
    main()