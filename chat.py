import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-1.7B"


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cpu"],
        default="auto",
    )
    args = parser.parse_args()

    # MacのGPUが使えるならMPS、それ以外はCPU
    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPSが利用できません。--device cpu で起動してください。"
        )

    # MacのGPUではFP16、CPUではFP32
    dtype = torch.float16 if device == "mps" else torch.float32

    print(f"モデルを読み込み中：{MODEL_ID}")
    print(f"計算先：{device} / 精度：{dtype}", flush=True)

    # ローカルキャッシュだけを使用する
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

    messages = []

    print("\n準備完了。質問を入力してください。")
    print("/reset：会話履歴を消す")
    print("/quit ：終了\n")

    while True:
        try:
            question = input("あなた > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if question.lower() == "/quit":
            break

        if question.lower() == "/reset":
            messages.clear()
            print("会話履歴を消しました。\n")
            continue

        if not question:
            continue

        # 今回の質問を会話履歴に追加した入力を作る
        current_messages = messages + [
            {"role": "user", "content": question}
        ]

        # Qwen専用の会話書式へ変換
        text = tokenizer.apply_chat_template(
            current_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)

        # 最初の実験では長い会話を避ける
        if inputs["input_ids"].shape[1] > 4096:
            print(
                "入力と会話履歴が長くなっています。"
                "/reset で履歴を消してください。\n"
            )
            continue

        print("生成中…", flush=True)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        # 入力部分を除き、新しく生成した部分だけ取り出す
        input_length = inputs["input_ids"].shape[1]
        answer_ids = output_ids[0, input_length:]

        answer = tokenizer.decode(
            answer_ids.tolist(),
            skip_special_tokens=True,
        ).strip()

        print(f"\nモデル > {answer}\n")

        # 次の質問でも、この会話を参照できるようにする
        messages = current_messages + [
            {"role": "assistant", "content": answer}
        ]


if __name__ == "__main__":
    main()