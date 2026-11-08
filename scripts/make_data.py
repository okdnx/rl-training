# scripts/make_data.py
import hashlib
import json
import random
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SEED = 42

SPLIT_SIZES = {
    "sft_train": 1000,
    "rl_train": 1000,
    "valid": 100,
    "test": 200,
}

OPERATIONS = ["+", "-", "*", "/"]


def calculate(left, operator, right):
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator == "/":
        if right == 0:
            raise ZeroDivisionError
        return left / right
    raise ValueError(operator)


def make_prompt(numbers, target):
    numbers_text = ", ".join(map(str, numbers))
    return (
        f"Use each number exactly once: {numbers_text}.\n"
        f"Make {target} using +, -, *, / and parentheses.\n"
        'Return only JSON with an "expression" field.'
    )


def make_example(rng):
    numbers = [rng.randint(2, 20) for _ in range(3)]

    # 表示順とは別の順番で正解の式を組み立てる
    a, b, c = rng.sample(numbers, len(numbers))
    op1 = rng.choice(OPERATIONS)
    op2 = rng.choice(OPERATIONS)

    if rng.choice([True, False]):
        value = calculate(
            calculate(Fraction(a), op1, Fraction(b)),
            op2,
            Fraction(c),
        )
        expression = f"({a} {op1} {b}) {op2} {c}"
    else:
        value = calculate(
            Fraction(a),
            op1,
            calculate(Fraction(b), op2, Fraction(c)),
        )
        expression = f"{a} {op1} ({b} {op2} {c})"

    # 最初は正の整数を目標にする
    if value.denominator != 1 or not 1 <= value <= 100:
        return None

    target = int(value)

    # 数字の並べ替えも同じ問題として扱う
    key = (tuple(sorted(numbers)), target)

    return key, {
        "numbers": numbers,
        "target": target,
        "prompt": make_prompt(numbers, target),
        "solution": expression,
        "completion": json.dumps({"expression": expression}),
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    filenames = [
        DATA_DIR / f"{name}.jsonl"
        for name in SPLIT_SIZES
    ]
    manifest_path = DATA_DIR / "manifest.json"

    # 既存の実験データを上書きしない
    for path in filenames + [manifest_path]:
        if path.exists():
            raise FileExistsError(
                f"{path} が既にあります。既存データは変更しません。"
            )

    rng = random.Random(SEED)
    examples = []
    seen = set()
    total = sum(SPLIT_SIZES.values())

    for _ in range(500_000):
        if len(examples) >= total:
            break

        try:
            result = make_example(rng)
        except ZeroDivisionError:
            continue

        if result is None:
            continue

        key, example = result
        if key in seen:
            continue

        seen.add(key)
        examples.append(example)

    if len(examples) < total:
        raise RuntimeError("必要な件数を生成できませんでした")

    rng.shuffle(examples)

    manifest = {
        "seed": SEED,
        "numbers_per_problem": 3,
        "number_range": [2, 20],
        "target_range": [1, 100],
        "splits": {},
    }

    offset = 0

    for name, count in SPLIT_SIZES.items():
        rows = examples[offset:offset + count]
        offset += count
        path = DATA_DIR / f"{name}.jsonl"

        with path.open("w", encoding="utf-8") as file:
            for index, example in enumerate(rows):
                row = {"id": f"{name}-{index:04d}", **example}

                # RL用には正解回答を保存しない
                if name == "rl_train":
                    del row["solution"]
                    del row["completion"]

                file.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )

        manifest["splits"][name] = {
            "count": count,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(f"{name}: {count}件 → {path}")

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n完了。学習はまだ実行していません。")


if __name__ == "__main__":
    main()