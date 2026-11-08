# scripts/inspect_rollouts.py
import argparse
import json
from datetime import datetime
from pathlib import Path

from scripts.generate import load_model, generate_answer
from scripts.verifier import verify_answer


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="checkpoints/sft-100",
    )
    parser.add_argument("--problems", type=int, default=10)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cpu", "cuda"],
        default="auto",
    )
    args = parser.parse_args()

    if args.problems < 1 or args.generations < 2:
        parser.error("problemsは1以上、generationsは2以上が必要です")

    adapter = Path(args.adapter).expanduser()
    if not adapter.is_absolute():
        adapter = ROOT / adapter

    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"LoRAの設定がありません：{adapter}"
        )

    # 検証用ではなく、これからRLに使う問題を読む
    data_path = ROOT / "data" / "rl_train.jsonl"

    with data_path.open(encoding="utf-8") as file:
        rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ][:args.problems]

    if not rows:
        raise ValueError("問題データが空です")

    model, tokenizer = load_model(
        device=args.device,
        adapter=adapter,
    )

    groups = []

    for question_index, row in enumerate(rows):
        print(
            f"\n── 問題 {question_index + 1}/{len(rows)} ──"
        )
        print(f"数字：{row['numbers']} / 目標：{row['target']}")

        answers = []

        for sample_index in range(args.generations):
            # 回答ごとに異なるseedを使う
            seed = 42 + (
                question_index * args.generations
                + sample_index
            )

            generated = generate_answer(
                model=model,
                tokenizer=tokenizer,
                prompt=row["prompt"],
                max_new_tokens=128,
                seed=seed,
            )

            grade = verify_answer(
                answer=generated["answer"],
                numbers=row["numbers"],
                target=row["target"],
            )

            answers.append({
                "seed": seed,
                **generated,
                **grade,
            })

            print(
                f"\n回答{sample_index + 1}："
                f"報酬={grade['reward']} / "
                f"{grade['reason']}"
            )
            print(generated["answer"], flush=True)

        rewards = [answer["reward"] for answer in answers]

        # 報酬が同じなら、相対的な優劣がない
        mixed = len(set(rewards)) > 1

        print(f"\n報酬の組：{rewards}")
        print(f"報酬に優劣あり：{mixed}", flush=True)

        groups.append({
            "id": row["id"],
            "numbers": row["numbers"],
            "target": row["target"],
            "prompt": row["prompt"],
            "rewards": rewards,
            "mixed": mixed,
            "answers": answers,
        })

    mixed_count = sum(group["mixed"] for group in groups)
    all_zero = sum(
        all(reward == 0 for reward in group["rewards"])
        for group in groups
    )
    all_one = sum(
        all(reward == 1 for reward in group["rewards"])
        for group in groups
    )

    summary = {
        "problems": len(groups),
        "generations_per_problem": args.generations,
        "mixed_groups": mixed_count,
        "all_zero_groups": all_zero,
        "all_one_groups": all_one,
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_path = results_dir / f"rollouts-{timestamp}.json"

    output_path.write_text(
        json.dumps(
            {
                "adapter": str(adapter),
                "device": str(model.device),
                "summary": summary,
                "groups": groups,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n── 集計 ──")
    print(f"報酬に優劣がある問題：{mixed_count}/{len(groups)}")
    print(f"全回答が不正解の問題：{all_zero}/{len(groups)}")
    print(f"全回答が正解の問題：{all_one}/{len(groups)}")
    print(f"保存先：{output_path}")


if __name__ == "__main__":
    main()