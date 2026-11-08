# profile_inference.py
import argparse
import json
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-1.7B"
ROOT = Path(__file__).resolve().parent

PROMPT = (
    "Explain supervised fine-tuning in simple English. "
    "Describe its purpose, training data, and how weights are updated."
)


def synchronize(device):
    # GPUの処理完了を待ち、時間を正しく測る
    if device == "mps":
        torch.mps.synchronize()


def memory_snapshot(device):
    # 測定時点の使用量。推論中の最大値ではない
    if device != "mps":
        return None

    return {
        "tensor_memory_mib": (
            torch.mps.current_allocated_memory() / 1024**2
        ),
        "driver_memory_mib": (
            torch.mps.driver_allocated_memory() / 1024**2
        ),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cpu"],
        default="auto",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()

    if args.runs < 1 or args.tokens < 1:
        parser.error("--runsと--tokensは1以上にしてください")

    if args.device == "auto":
        device = (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = args.device

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPSが利用できません。--device cpu を指定してください。"
        )

    dtype = (
        torch.float16 if device == "mps" else torch.float32
    )

    print(f"モデル：{MODEL_ID}")
    print(f"計算先：{device} / 精度：{dtype}")
    print("モデルを読み込んでいます…", flush=True)

    synchronize(device)
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        dtype=dtype,
        device_map=device,
        attn_implementation="sdpa",
    )
    model.eval()
    model.requires_grad_(False)

    synchronize(device)
    load_seconds = time.perf_counter() - started
    memory_after_load = memory_snapshot(device)

    # 会話履歴なし・毎回同じ質問を使う
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    input_tokens = inputs["input_ids"].shape[1]

    def generate_once():
        # 毎回同じ乱数seedで測定する
        torch.manual_seed(42)

        synchronize(device)
        start = time.perf_counter()

        outputs = model.generate(
            **inputs,
            max_new_tokens=args.tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        synchronize(device)
        elapsed = time.perf_counter() - start

        # 文字列への変換は生成時間に含めない
        generated_ids = outputs[0, input_tokens:].tolist()
        answer = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return {
            "seconds": elapsed,
            "output_tokens": len(generated_ids),
            "output_tokens_per_second": (
                len(generated_ids) / elapsed
            ),
            "memory_after_generation": memory_snapshot(device),
            "answer": answer,
        }

    print(f"読み込み：{load_seconds:.2f}秒")
    print(f"入力：{input_tokens}トークン")
    print("\nウォームアップを1回実行…", flush=True)

    warmup = generate_once()

    print(f"ウォームアップ：{warmup['seconds']:.2f}秒")
    print(f"\n本測定を{args.runs}回実行します。", flush=True)

    runs = []

    for index in range(args.runs):
        result = generate_once()
        runs.append(result)

        print(
            f"{index + 1}回目："
            f"{result['seconds']:.2f}秒 / "
            f"{result['output_tokens']}トークン / "
            f"{result['output_tokens_per_second']:.2f} tok/s",
            flush=True,
        )

    summary = {
        "median_seconds": statistics.median(
            run["seconds"] for run in runs
        ),
        "median_output_tokens_per_second": statistics.median(
            run["output_tokens_per_second"] for run in runs
        ),
    }

    report = {
        "model": MODEL_ID,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "device": device,
        "dtype": str(dtype),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "prompt": PROMPT,
        "input_tokens": input_tokens,
        "generation": {
            "max_new_tokens": args.tokens,
            "seed": 42,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "thinking": False,
            "use_cache": True,
        },
        "load_seconds": load_seconds,
        "memory_after_load": memory_after_load,
        "warmup": warmup,
        "runs": runs,
        "summary": summary,
    }

    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_path = output_dir / f"profile-{device}-{timestamp}.json"

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n--- 測定結果 ---")
    print(f"生成時間の中央値：{summary['median_seconds']:.2f}秒")
    print(
        "生成速度の中央値："
        f"{summary['median_output_tokens_per_second']:.2f} tok/s"
    )
    print(f"保存先：{output_path}")
    print("\n--- 生成された回答の例 ---")
    print(runs[0]["answer"])


if __name__ == "__main__":
    main()