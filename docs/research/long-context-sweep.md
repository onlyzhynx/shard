# Long-context FastVerify sweep

## Why this exists

Shard's fast verify path stores the full request KV in `StaticKV`, then captures
decode graphs against context buckets. Once a request grows beyond 16,384 tokens,
decode moves into the 32,768 bucket. Full-KV attention can therefore produce a
visible `verify_ms` increase and `tok_s` drop around that boundary.

`FV_WINDOW=1` is an existing opt-in path that lets sliding-attention layers read
only their configured window. It can reduce long-context verify work, but the
different attention tiling can introduce small floating-point differences that
occasionally affect MoE routing. This benchmark measures both the speed benefit
and exact output-token divergence before any cache-policy change is considered.
It does not enable `FV_WINDOW` for normal inference and does not implement an
approximate mode, sink-plus-recent cache, virtual context, or KV-page retrieval.

## What it measures

`research/context_sweep.py` loads the requested target as a single local stage,
constructs deterministic token contexts, prefills `FastVerify`, and performs
fixed-`K` greedy verification for a small number of new tokens. It needs one CUDA
GPU and no WAN swarm. The complete target model must fit on that GPU, so use a
representative model that fits locally when the production target is sharded.

The synthetic prompt is exactly the requested token length. It repeats a varied,
neutral prose token sequence instead of one token. Candidate draft tokens are also
deterministic and deliberately cheap: the harness is intended to isolate local
`FastVerify` and `StaticKV` behavior, not draft-model quality. Greedy target
correction remains authoritative, so baseline and window-mode output IDs can be
compared exactly. Generation intentionally continues through EOS IDs so every run
measures the same requested `--max-new` token budget.

For every mode and context, the JSONL contains optional warmup run records,
measured run records, and one `record_type="summary"` record. Summary metrics are
means over `--repeat` measured runs only. Raw `output_ids` are retained so another
run can use the file as an exact baseline.

Important fields are:

- `tok_s`: generated target tokens per measured decode second; prefill is excluded.
- `verify_ms`: mean local verification time per traversal, including the target
  decode, final norm, LM head, argmax, and CUDA synchronization.
- `draft_ms`: mean time spent constructing deterministic candidates. This should
  be near zero and is present for schema compatibility with speculative runs.
- `tokens_per_traversal` and `mean_accept`: verify efficiency for the deterministic
  candidates. They provide context, but are not draft-quality measurements.
- `gpu_memory_allocated_mb` and `gpu_memory_reserved_mb`: CUDA memory after a run.
- `output_ids_match_baseline`: exact integer-list equality, never decoded-text
  equality. `first_differing_token_index`, `baseline_output_ids_sha256`, and
  `test_output_ids_sha256` locate and identify any divergence.
- `output_ids_consistent`: whether all measured repeats in a summary produced the
  same output IDs.

## Run the comparison

The default mode order is `baseline` followed by `fv_window`. Before constructing
the baseline `FastVerify`, the script removes `FV_WINDOW` from its environment.
Before constructing the comparison instance, it sets `FV_WINDOW=1`. The process
removes the variable again when the sweep completes.

```powershell
python research/context_sweep.py `
  --model <MODEL_ID> `
  --ctx 2048 4096 8192 16000 20000 24000 32768 `
  --max-new 64 `
  --repeat 3 `
  --out results/context_sweep_baseline_vs_fvwindow.jsonl
```

On shells that use backslash continuation:

```bash
python research/context_sweep.py \
  --model <MODEL_ID> \
  --ctx 2048 4096 8192 16000 20000 24000 32768 \
  --max-new 64 \
  --repeat 3 \
  --out results/context_sweep_baseline_vs_fvwindow.jsonl
```

The full default context list also includes 12,000 tokens. Unless `--max-ctx` is
provided, the script sizes it to the largest context plus `--max-new` and `K`
scratch positions. Set it explicitly to compare runs with the same cache ceiling:

```bash
python research/context_sweep.py --model <MODEL_ID> --max-ctx 33000 --K 6
```

To run modes separately, first produce a baseline file, then point the window run
at it so exact IDs remain available:

```bash
python research/context_sweep.py --model <MODEL_ID> --mode baseline \
  --out results/context_sweep_baseline.jsonl
python research/context_sweep.py --model <MODEL_ID> --mode fv_window \
  --baseline-file results/context_sweep_baseline.jsonl \
  --out results/context_sweep_fv_window.jsonl
```

Warmups default to one run and are logged with `warmup=true`; they never contribute
to summaries. Use `--no-log-warmup` to omit them. The default measured repeat count
is three.

## Interpreting a useful result

A useful sweep has stable measured repeats, a clear `verify_ms` and `tok_s` curve
across 16,000, 20,000, 24,000, and 32,768 tokens, and the same model, `max_ctx`,
`K`, device, prompt construction, and generation length in both modes. The main
question is whether `fv_window` reduces the post-16k verify cliff enough to matter.

Speedup is actionable only alongside output evidence. Exact matches at every
context support considering `FV_WINDOW` as a safe opt-in long-context lever for
that model and hardware stack. Any mismatch should be reported with its first
differing token and both hashes; even a rare mismatch means the path cannot be
described as generally bit-exact or made the default without a separate policy
decision.

The utility tests do not import CUDA dependencies:

```bash
python -m unittest discover -s tests -p "test_context_sweep_utils.py"
```
