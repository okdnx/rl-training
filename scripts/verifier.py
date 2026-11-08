# scripts/verifier.py
import argparse
import ast
import json
from collections import Counter
from fractions import Fraction


ALLOWED_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div)


def reject_duplicate_keys(pairs):
    """同じJSONキーの重複を認めない。"""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def collect_numbers(node):
    """許可された式か確認し、使われた整数を集める。"""
    if isinstance(node, ast.Constant):
        # boolや小数は受け付けない
        if type(node.value) is not int:
            raise ValueError("Only integer literals are allowed")
        return [node.value]

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, ALLOWED_OPERATORS):
            raise ValueError("Only +, -, *, / are allowed")

        return (
            collect_numbers(node.left)
            + collect_numbers(node.right)
        )

    # 関数呼び出し・変数・単項演算などは認めない
    raise ValueError("Unsupported expression")


def calculate(node):
    """浮動小数点の誤差を避け、分数として計算する。"""
    if isinstance(node, ast.Constant):
        return Fraction(node.value)

    left = calculate(node.left)
    right = calculate(node.right)

    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Div):
        if right == 0:
            raise ZeroDivisionError
        return left / right

    raise ValueError("Unsupported operator")


def verify_answer(answer, numbers, target):
    """
    answer: モデルが生成した文字列
    numbers: 指定した整数のリスト
    target: 目標の整数
    """
    if (
        not numbers
        or len(numbers) > 8
        or any(
            type(number) is not int
            or not 1 <= number <= 1_000_000
            for number in numbers
        )
        or type(target) is not int
    ):
        raise ValueError("Invalid problem data")

    result = {
        "reward": 0.0,
        "correct": False,
        "format_ok": False,
        "numbers_ok": False,
        "target_ok": False,
        "expression": None,
        "value": None,
        "reason": "",
    }

    def fail(reason):
        result["reason"] = reason
        return result

    if not isinstance(answer, str) or len(answer) > 8192:
        return fail("invalid_answer_length")

    try:
        data = json.loads(
            answer,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (ValueError, RecursionError):
        return fail("invalid_json")

    # JSONオブジェクトで、キーはexpressionだけ
    if not isinstance(data, dict) or set(data) != {"expression"}:
        return fail("invalid_json_schema")

    expression = data["expression"]

    if not isinstance(expression, str):
        return fail("expression_must_be_string")

    expression = expression.strip()

    if not expression or len(expression) > 256:
        return fail("invalid_expression_length")

    result["format_ok"] = True
    result["expression"] = expression

    try:
        tree = ast.parse(expression, mode="eval")

        if sum(1 for _ in ast.walk(tree)) > 64:
            return fail("expression_too_complex")

        used_numbers = collect_numbers(tree.body)
    except (SyntaxError, ValueError, RecursionError):
        return fail("invalid_expression")

    # 重複する数も個数まで一致させる
    # 例：[2, 2, 8]なら、2を2回・8を1回使う
    if Counter(used_numbers) != Counter(numbers):
        return fail("wrong_numbers")

    result["numbers_ok"] = True

    try:
        value = calculate(tree.body)
    except ZeroDivisionError:
        return fail("division_by_zero")

    result["value"] = str(value)
    result["target_ok"] = value == Fraction(target)

    if not result["target_ok"]:
        return fail("wrong_target")

    result["reward"] = 1.0
    result["correct"] = True
    result["reason"] = "correct"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="数字から作った式の回答を採点"
    )
    parser.add_argument(
        "--numbers",
        nargs="+",
        type=int,
        required=True,
    )
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--answer", required=True)
    args = parser.parse_args()

    result = verify_answer(
        answer=args.answer,
        numbers=args.numbers,
        target=args.target,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()