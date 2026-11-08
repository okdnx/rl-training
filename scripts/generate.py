import argparse
import json

from peft import PeftModel 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-1.7B"


def load_model(device="auto", adapter=None):
    """モデルとトークナイザーを読み込む。"""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDAが利用できません")

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPSが利用できません")

    if device == "cuda":
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(
        f"読み込み中：{MODEL_ID} / {device} / {dtype}",
        flush=True,
    )

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

    if adapter is not None:
        model = PeftModel.from_pretrained(
            model,
            str(adapter),
            is_trainable=False,
        )

    model.eval()
    model.requires_grad_(False)

    return model, tokenizer


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    prompt,
    max_new_tokens=128,
    seed=42,
):
    """1つの質問から回答を生成する。会話履歴は持たない。"""
    torch.manual_seed(seed)

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    input_length = inputs["input_ids"].shape[1]
    answer_ids = outputs[0, input_length:].tolist()

    return {
        "answer": tokenizer.decode(
            answer_ids,
            skip_special_tokens=True,
        ).strip(),
        "input_tokens": input_length,
        "output_tokens": len(answer_ids),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default=(
            "Use each number exactly once: 8, 2, 4.\n"
            "Make 24 using +, -, *, / and parentheses.\n"
            'Return only JSON with an "expression" field.'
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.device)

    result = generate_answer(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
    )

    print("\n質問：")
    print(args.prompt)
    print("\n生成結果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()