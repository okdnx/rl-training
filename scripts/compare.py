# scripts/compare.py
import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_report(filename):
    path = Path(filename).expanduser()

    if not path.is_absolute():
        path = ROOT / path

    with path.open(encoding="utf-8") as file:
        report = json.load(file)

    records = {}

    for record in report["records"]:
        question_id = record["id"]

        if question_id in records:
            raise ValueError(f"重複した問題ID：{question_id}")

        records[question_id] = record

    return path, report, records


def main():
    parser = argparse.ArgumentParser(
        description="学習前後の回答を問題ごとに比較する"
    )
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument(
        "--show",
        type=int,
        default=5,
        help="改善・悪化した問題をそれぞれ何件表示するか",
    )
    args = parser.parse_args()

    if args.show < 0:
        parser.error("--showは0以上にしてください")

    before_path, before_report, before = read_report(args.before)
    after_path, after_report, after = read_report(args.after)

    if set(before) != set(after):
        raise ValueError(
            "評価した問題IDが一致しません。"
            "同じsplit・同じ件数の結果を指定してください。"
        )

    if not before:
        raise ValueError("比較する問題がありません")

    # 比較条件を確認
    for key in [
        "model",
        "split",
        "data_sha256",
        "generation_settings",
        "device",
        "dtype",
    ]:
        if before_report.get(key) != after_report.get(key):
            raise ValueError(
                f"比較条件が異なります：{key}\n"
                f"学習前：{before_report.get(key)}\n"
                f"学習後：{after_report.get(key)}"
            )

    groups = {
        "both_correct": [],
        "improved": [],
        "regressed": [],
        "both_wrong": [],
    }

    for question_id, old in before.items():
        new = after[question_id]

        for key in ["prompt", "numbers", "target"]:
            if old[key] != new[key]:
                raise ValueError(
                    f"{question_id}の問題内容が異なります：{key}"
                )

        if old["correct"] and new["correct"]:
            group = "both_correct"
        elif not old["correct"] and new["correct"]:
            group = "improved"
        elif old["correct"] and not new["correct"]:
            group = "regressed"
        else:
            group = "both_wrong"

        groups[group].append({
            "id": question_id,
            "numbers": old["numbers"],
            "target": old["target"],
            "before_answer": old["answer"],
            "after_answer": new["answer"],
            "before_reason": old["reason"],
            "after_reason": new["reason"],
        })

    count = len(before)
    before_correct = sum(
        row["correct"] for row in before.values()
    )
    after_correct = sum(
        row["correct"] for row in after.values()
    )

    print("── 全体比較 ──")
    print(f"問題数：{count}")
    print(
        f"学習前：{before_correct}/{count} "
        f"({before_correct / count:.1%})"
    )
    print(
        f"学習後：{after_correct}/{count} "
        f"({after_correct / count:.1%})"
    )

    labels = {
        "both_correct": "両方正解",
        "improved": "改善：不正解 → 正解",
        "regressed": "悪化：正解 → 不正解",
        "both_wrong": "両方不正解",
    }

    print("\n── 問題ごとの変化 ──")
    for group, label in labels.items():
        print(f"{label}：{len(groups[group])}問")

    print("\n── 採点理由の比較 ──")
    before_reasons = Counter(
        row["reason"] for row in before.values()
    )
    after_reasons = Counter(
        row["reason"] for row in after.values()
    )

    for reason in sorted(set(before_reasons) | set(after_reasons)):
        print(
            f"{reason}: "
            f"{before_reasons[reason]} → {after_reasons[reason]}"
        )

    for group in ["regressed", "improved"]:
        print(f"\n── {labels[group]}の例 ──")

        for row in groups[group][:args.show]:
            print(
                f"\n{row['id']} / "
                f"数字={row['numbers']} / "
                f"目標={row['target']}"
            )
            print(
                f"学習前 [{row['before_reason']}]: "
                f"{row['before_answer']}"
            )
            print(
                f"学習後 [{row['after_reason']}]: "
                f"{row['after_answer']}"
            )

    output_path = (
        after_path.parent / f"{after_path.stem}-comparison.json"
    )
    output_path.write_text(
        json.dumps(
            {
                "before": str(before_path),
                "after": str(after_path),
                "count": count,
                "groups": groups,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n比較結果の保存先：{output_path}")


if __name__ == "__main__":
    main()