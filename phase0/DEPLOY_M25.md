# MiniMax-M2.5 — single disciplined GPU validation pass

The engine is **code-complete and locally proven (no-GPU)**: tool calling, multi-turn context,
signed per-stage receipts, and the OpenAI `/v1` gateway all pass local tests
(`research/m25_{tools,gateway,receipt}_test.py`, 47 assertions). The ONLY thing that needs GPUs is
the warm-libp2p validation + the CUDA-graph perf lever. This runbook makes that pass mechanical, so
it does NOT repeat the morning-killers (stuck downloads, `pkill` self-match, blind 30-min waits).

## Hard ops rules (every step)
- **No blind waits.** Every download/launch runs under a hard per-phase deadline. Stuck > deadline →
  kill + replace the box, never sit on the shell.
- **Over-provision 6-for-5.** Rent 6 boxes for a 5-stage ring; drop the slowest/flakiest.
- **`setsid … </dev/null &` + `fuser -k <port>/tcp`. NEVER `pkill -f`** (it self-matches the launch
  string and kills the launcher — the documented footgun).
- **Robust precheck before bootstrapping a box:** SSH-retry + `urllib` HF-reachability (NOT `curl`,
  not preinstalled) + GPU-count. Some vast hosts DNS-hijack huggingface.co — apt/pip work, HF doesn't.
- **Verify every file push** (grep a known line) before relying on it. scp inside `( … ) &` can
  silently not land.

## Topology
5 scattered US 5090s, even ~12-13 layers/stage over 62L, direct-return pipeline (head fire-forwards,
tail returns to coord). Sidecar binary at `/tmp/sidecar` (prebuilt June-19; can't rebuild on go1.22).
Always create boxes with `--env '-p 29600:29600'` (inter-stage transport unreachable otherwise).

## Sequence (driven by `m25_scatter_pipe.py`)
1. **Precheck** each candidate box: `ssh` reachable, `urllib` GET on the HF tokenizer_config 200, GPU
   count == expected. Drop failures, pull from the over-provision pool.
2. **Bootstrap** (per box, deadline ~12 min): venv + `pip install vllm` (→ vLLM 0.23 + torch/cu13 +
   flashinfer, just works on sm_120) + push code. Push set now includes **`m25_tools.py`** (hard dep
   of `m25_pipe`) and **`receipt.py` + `manifest.py`** (so `SHARD_RECEIPTS=1` actually loads).
3. **Pull layer-range shards** (deadline ~15 min, hf_transfer; fallback `HF_HUB_ENABLE_HF_TRANSFER=0`
   if it STALLS): `m25_pull_range.py --lo L --hi H` per stage; `--head` adds embed+tokenizer, `--tail`
   adds norm+lm_head. **Verify** each box reports the expected shard count before launching.
4. **Sidecars** then **stages** (tail-first), each launched `setsid`, health-grepped (`tunnel up|
   listening` for sidecar, `WARM` for stage), retried, never `pkill`ed.
5. **Coordinator / gateway** on the head box.

## Validation (what the pass must prove, WARM over libp2p)
- **tok/s**: copy task, depth-4 pipeline. Baseline 15.79; with the CUDA-graph lever target ~20-25+.
  `m25_pipe.py coord --head … --tail … --K 6 --depth 4 --prompt-file copy.txt`
- **Tool calling**: serve the gateway, POST `/v1/chat/completions` with `tools=[…]`, assert
  `finish_reason=="tool_calls"` and a structured `tool_calls[0].function`. (Parser already proven
  locally against the real tokenizer; this confirms the model emits the format end-to-end.)
- **Multi-turn context**: 2-3 turn conversation incl. a tool result; long-context prefill (≥30k) for
  the pipelined-prefill number.
- **Receipts**: `SHARD_RECEIPTS=1` on every stage + coord. Coord prints N signed receipts, all sigs
  VALID, coverage `[0:62]` no gap/overlap. (`x_shard.receipts_ok` in the gateway response.)

## CUDA-graph lever (#6 — develop ON the box, it's empirical)
The per-traversal ~95ms GPU is launch-overhead-bound (19.7ms/stage, FLAT in token count). To capture:
- Replace the grow-by-`cat` KV cache in `m25_stage.Layer.attn` (lines ~144-149) with a **pre-allocated
  static KV buffer** `[1, NKV, MAXLEN, HD]`, in-place write at `start_pos`, fixed-shape masked
  attention over `[:cur_len]`. Pass `start_pos`/`cur_len` via small static input buffers updated
  before each replay.
- Capture a CUDA graph of `run_block` for the **s=1 decode shape**, replay per step. Validate the
  NVFP4 cutlass FusedMoE is graph-safe (vLLM graphs it internally — expected OK, confirm).
- Keep the eager path as fallback (`M25_CUDA_GRAPH=0`). Prefill stays eager.
- Verify bit-equivalence vs eager (greedy ids identical) before trusting the number.

## Privacy posture (already true, state it; don't over-claim)
- libp2p transport is Noise-encrypted node-to-node by default — no PSK, per-node keys.
- **Intermediate stages only ever see hidden-state tensors, never tokens/text.** Only the head sees
  input token ids; only the tail produces output tokens. So no single middle node can reconstruct the
  prompt or the answer. Stronger guarantees (coordinator-blind prompts, activation obfuscation) are
  research-grade, out of scope for the beta.
