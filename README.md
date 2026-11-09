# Qwen inference profiling & post-training experiments

A compact LLM experiment pipeline built around **Qwen3-1.7B**, with single-request inference profiling, LoRA supervised fine-tuning, GRPO, and verifiable arithmetic evaluation.

The repository includes the experiment code, data manifests, adapters, and per-question results. Start with the measurements below, then follow the linked code to inspect how they were produced.

## Recorded results

### Inference baseline

The [saved MPS report](results/profile-mps-20261107-221624-192129.json) records a **31.56 output tok/s median** across three measured runs, after one warm-up run.

| Setting | Recorded value |
| --- | --- |
| Model | `Qwen/Qwen3-1.7B` |
| Backend / dtype | Apple Silicon MPS / FP16 |
| Input / generated tokens | 35 / 128 per run |
| Median generation time | 4.06 s |
| Median output tokens / generation time | 31.56 tok/s |
| Model + tokenizer load time | 3.58 s |
| Tensor / driver memory after loading | 3296.24 / 4296.69 MiB |

[Profiler implementation](profile_inference.py): device synchronization surrounds each timed generation; decoding to text happens afterward. Generation uses SDPA, KV caching, sampling, and a fixed seed. Memory figures are snapshots, not peak usage.

This is end-to-end `generate()` timing for one prompt at a time. It does not separate prefill from decode, measure time to first token, or measure concurrent serving throughput. The report does not identify the Apple chip model, so it should not be used for hardware-to-hardware comparisons.

### Arithmetic evaluation

All three reports use the same 200-question test split, MPS FP16, generation settings, and per-question seed rule.

| Model variant | Correct answers | Accuracy | Required numbers used correctly |
| --- | ---: | ---: | ---: |
| [Base Qwen3-1.7B](results/base-test-20261108-094359-351993.json) | 52 / 200 | 26.0% | 82.0% |
| [LoRA SFT, 100 steps](results/sft-100-test-20261108-094708-125712.json) | 75 / 200 | 37.5% | 99.5% |
| [SFT + GRPO, 100 additional steps](results/grpo-100-test-20261108-095136-942616.json) | 79 / 200 | 39.5% | 100.0% |

The final adapter improves accuracy by **13.5 percentage points over the base model**, and 2.0 points over SFT. In the base-to-GRPO paired comparison, 49 answers improve and 22 regress. These are single-run results on a small synthetic task; they do not establish broad reasoning improvements or statistical significance.

## What to inspect

| Area | Implementation | Design choice |
| --- | --- | --- |
| Inference measurement | [profile_inference.py](profile_inference.py) | Warm-up, synchronization, repeated runs, saved environment metadata |
| Generation | [scripts/generate.py](scripts/generate.py) | Shared prompt template and generation settings; optional LoRA adapter |
| Evaluation | [scripts/evaluate.py](scripts/evaluate.py) | Per-question records, dataset SHA-256, model/device metadata, failure reasons |
| Result comparison | [scripts/compare.py](scripts/compare.py) | Rejects mismatched datasets, settings, devices, or question IDs |
| Verifiable reward | [scripts/verifier.py](scripts/verifier.py) | Restricted AST parsing and exact rational arithmetic; no Python `eval()` |
| Post-training | [SFT](scripts/train_sft.py) / [GRPO](scripts/train_grpo.py) | LoRA adapters and an arithmetic correctness reward |
| CUDA environment | [runpod/](runpod/) | Separate locked dependencies and a digest-pinned container definition |

The verifier requires a JSON object with exactly one `expression` field, allows only integer literals and `+ - * /`, checks the multiset of supplied numbers, and compares the result using `Fraction`.

## Inspect results without downloading a model

Python's standard library is sufficient for the verifier and report comparison:

```bash
git clone https://github.com/okdnx/rl-training.git
cd rl-training

python3 scripts/verifier.py \
  --numbers 8 2 4 --target 24 \
  --answer '{"expression":"(8 + 4) * 2"}'

python3 scripts/compare.py \
  --before results/base-test-20261108-094359-351993.json \
  --after results/grpo-100-test-20261108-095136-942616.json \
  --show 3
```

## Run inference locally

Requirements: Python 3.14 and [uv](https://docs.astral.sh/uv/). The root environment is the starting point for Apple Silicon; CUDA dependencies are defined separately in [runpod/pyproject.toml](runpod/pyproject.toml).

```bash
uv sync --locked

# The scripts load local files only; populate the model cache first.
uv run hf download Qwen/Qwen3-1.7B

# One warm-up, then three measured generations.
uv run python profile_inference.py --device mps --runs 3 --tokens 128

# Evaluate the base model and the included adapter on the same test split.
uv run python -m scripts.evaluate --split test --limit 200 --device mps
uv run python -m scripts.evaluate --split test --limit 200 --device mps \
  --adapter checkpoints/grpo-100
```

Use `--device cpu` for a CPU run; its FP32 results are a different measurement condition. The profiler supports MPS and CPU. The shared generation/evaluation code also supports CUDA, but this repository's saved inference baseline is MPS.

The profiler and evaluator create timestamped JSON files under `results/`. Running inference downloads model weights and requires substantially more memory than inspecting the saved reports.

## Experiment data and training

The arithmetic task uses three integers in the range 2–20 and a target in the range 1–100. Each supplied number must appear exactly once in an expression that reaches the target.

| Split | Examples |
| --- | ---: |
| SFT training | 1000 |
| RL training | 1000 |
| Validation | 100 |
| Test | 200 |

[The manifest](data/manifest.json) records generation seed 42 and the SHA-256 of every split. [The generator](scripts/make_data.py) deduplicates problems by their sorted numbers and target and refuses to overwrite existing data.

SFT uses rank-16 LoRA on `q_proj` and `v_proj`. GRPO starts from `checkpoints/sft-100` and uses the same verifier as evaluation. See the [training scripts](scripts/) and [saved training reports](results/) for settings and logs. Use a new output name when running training; existing checkpoint directories are protected from overwriting.

A [CUDA SFT training report](results/runpod-sft-100-training.json) records 100 updates in 92.63 seconds. This is a training measurement, not a CUDA inference benchmark; the report does not include enough GPU hardware detail for a performance comparison.

## Scope and next experiments

This project currently demonstrates local generation measurement, adapter training, and auditable evaluation. It does not yet contain a production serving system or measured CUDA-kernel optimizations.

Next experiments:

- Record exact accelerator hardware and pin model revisions; historical reports contain a null model revision.
- Separate prefill, time to first token, and inter-token latency.
- Measure latency distributions and throughput across prompt lengths and concurrency levels.
- Compare serving backends under the same model, hardware, and workload.
- Repeat quality evaluation across seeds and quantify uncertainty.

日本語: Qwen3-1.7Bの推論計測、LoRA SFT、GRPO、算術タスクの評価をまとめた実験リポジトリです。上記の数値は保存済みレポートに基づきます。
