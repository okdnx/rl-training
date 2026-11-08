# scripts/train_grpo.py
import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from scripts.generate import MODEL_ID
from scripts.verifier import verify_answer


ROOT = Path(__file__).resolve().parents[1]


def arithmetic_reward(completions, numbers, target, **kwargs):
    """生成された各回答を、既存の採点器で採点する。"""
    return [
        verify_answer(
            answer=answer,
            numbers=values,
            target=goal,
        )["reward"]
        for answer, values, goal in zip(
            completions, numbers, target
        )
    ]


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--name", default="grpo-pilot")
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--stepsは1以上にしてください")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        raise RuntimeError(
            "今回はMPSまたはCUDA GPUで実行してください"
        )

    sft_path = ROOT / "checkpoints" / "sft-100"
    output_path = ROOT / "checkpoints" / args.name

    if not (sft_path / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"SFTのLoRAがありません：{sft_path}"
        )

    if output_path.exists():
        raise FileExistsError(
            f"{output_path} が既にあります。"
            "--nameを変更してください"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"元モデルを読み込み中：{device}", flush=True)

    # 最初はSFT時と同じFP32を使う
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    base_model.to(device)

    # SFTで学習したLoRAを、更新できる状態で読み込む
    model = PeftModel.from_pretrained(
        base_model,
        str(sft_path),
        is_trainable=True,
    )
    model.config.use_cache = False

    model.print_trainable_parameters()

    data_path = ROOT / "data" / "rl_train.jsonl"

    with data_path.open(encoding="utf-8") as file:
        source_rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not source_rows:
        raise ValueError("RL用データが空です")

    rows = []

    for row in source_rows:
        # SFT・評価と同じ会話書式
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        rows.append({
            "prompt": prompt,
            "numbers": row["numbers"],
            "target": row["target"],
        })

    dataset = Dataset.from_list(rows)

    config = GRPOConfig(
        output_dir=str(output_path),
        max_steps=args.steps,

        learning_rate=5e-6,
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=0.0,

        # GPU1台で、1問題につき4回答を生成
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        steps_per_generation=4,
        num_generations=4,
        num_iterations=1,

        max_completion_length=128,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        generation_kwargs={"use_cache": True},

        # まず基本形のGRPO
        loss_type="grpo",
        scale_rewards="group",
        epsilon=0.2,

        # 初回は基準モデルへのKLペナルティなし
        beta=0.0,

        # 長さ制限で切れた回答は、更新対象から除く
        mask_truncated_completions=True,

        bf16=False,
        fp16=False,
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },
        max_grad_norm=1.0,

        use_vllm=False,
        logging_steps=1,
        save_strategy="no",

        seed=42,
        report_to="none",
        push_to_hub=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=arithmetic_reward,
    )

    synchronize(device)
    started = time.perf_counter()

    trainer.train()

    synchronize(device)
    elapsed = time.perf_counter() - started

    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    results_path = ROOT / "results"
    results_path.mkdir(exist_ok=True)

    report = {
        "base_model": MODEL_ID,
        "starting_adapter": str(sft_path),
        "saved_adapter": str(output_path),
        "device": device,
        "steps": trainer.state.global_step,
        "elapsed_seconds": elapsed,
        "versions": {
            name: version(name)
            for name in ["torch", "transformers", "trl", "peft"]
        },
        "history": trainer.state.log_history,
    }

    (results_path / f"{args.name}-training.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nRL学習時間：{elapsed:.1f}秒")
    print(f"保存先：{output_path}")


if __name__ == "__main__":
    main()