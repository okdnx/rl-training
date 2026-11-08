# scripts/evaluate.py
import argparse
import hashlib
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from scripts.generate import MODEL_ID, load_model, generate_answer
from scripts.verifier import verify_answer


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="元モデルまたはLoRA適用モデルの正解率を評価"
    )
    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        default="valid",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="評価する件数",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="LoRAのフォルダー。省略すると元モデルを評価",
    )
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limitは1以上にしてください")

    # 評価データを読み込む
    data_path = ROOT / "data" / f"{args.split}.jsonl"

    if not data_path.is_file():
        raise FileNotFoundError(
            f"{data_path} がありません。"
            "先にmake_data.pyでデータを生成してください。"
        )

    with data_path.open(encoding="utf-8") as file:
        all_rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    rows = all_rows[:args.limit]

    if not rows:
        raise ValueError("評価データが空です")

    # 相対パスはプロジェクトのフォルダーを基準にする
    adapter_path = None

    if args.adapter:
        adapter_path = Path(args.adapter).expanduser()

        if not adapter_path.is_absolute():
            adapter_path = ROOT / adapter_path

        adapter_path = adapter_path.resolve()

        if not adapter_path.is_dir():
            raise FileNotFoundError(
                f"LoRAフォルダーがありません：{adapter_path}"
            )

        if not (adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"adapter_config.jsonがありません：{adapter_path}"
            )

    print(f"評価対象：{adapter_path or MODEL_ID}")
    print(f"データ：{args.split} / {len(rows)}問\n")

    # モデルは最初に1回だけ読み込む
    model, tokenizer = load_model(
        device=args.device,
        adapter=adapter_path,
    )

    records = []

    for index, row in enumerate(rows):
        # 各問題を独立して入力。会話履歴は引き継がない
        generated = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=row["prompt"],
            max_new_tokens=128,
            seed=42 + index,
        )

        grade = verify_answer(
            answer=generated["answer"],
            numbers=row["numbers"],
            target=row["target"],
        )

        record = {
            "id": row["id"],
            "prompt": row["prompt"],
            "numbers": row["numbers"],
            "target": row["target"],
            **generated,
            **grade,
        }
        records.append(record)

        print(
            f"[{index + 1}/{len(rows)}] "
            f"correct={grade['correct']} "
            f"reason={grade['reason']}",
            flush=True,
        )
        print(f"回答：{generated['answer']}\n", flush=True)

    count = len(records)

    summary = {
        "count": count,
        "format_rate": sum(
            row["format_ok"] for row in records
        ) / count,
        "numbers_rate": sum(
            row["numbers_ok"] for row in records
        ) / count,
        "accuracy": sum(
            row["correct"] for row in records
        ) / count,
    }

    # 不正解の原因も集計
    reason_counts = {}

    for record in records:
        reason = record["reason"]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    report = {
        "model": MODEL_ID,
        "model_revision": getattr(
            model.config, "_commit_hash", None
        ),
        "adapter": str(adapter_path) if adapter_path else None,
        "device": str(model.device),
        "dtype": str(model.dtype),
        "versions": {
            name: version(name)
            for name in ["torch", "transformers", "peft"]
        },
        "split": args.split,
        "requested_limit": args.limit,
        "data_sha256": hashlib.sha256(
            data_path.read_bytes()
        ).hexdigest(),
        "generation_settings": {
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "thinking": False,
            "seed_rule": "42 + question_index",
        },
        "summary": summary,
        "reason_counts": reason_counts,
        "records": records,
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    label = adapter_path.name if adapter_path else "base"

    output_path = (
        results_dir / f"{label}-{args.split}-{timestamp}.json"
    )

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("── 集計 ──")
    print(f"問題数：{count}")
    print(f"JSON形式成功率：{summary['format_rate']:.1%}")
    print(f"数字の条件成功率：{summary['numbers_rate']:.1%}")
    print(f"最終正解率：{summary['accuracy']:.1%}")

    print("\n── 採点理由の内訳 ──")
    for reason, frequency in sorted(reason_counts.items()):
        print(f"{reason}: {frequency}")

    print(f"\n保存先：{output_path}")


if __name__ == "__main__":
    main()