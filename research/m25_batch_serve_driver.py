"""Drive coordinate_pipe_batch over a live 2-stage ring (head :29610, tail :29611) to validate the
batched SERVE protocol end-to-end. Three gates:
  1. COHERENCE   — B distinct prompts each return sensible text.
  2. DATA ISOLATION (the verification-critical gate) — stream A's output_ids are IDENTICAL whether its
     batch-mates are {B,C} or {X,Y}. Per-stream MoE + row-independent matmul => a stream's output depends
     only on its own data + the batch SIZE, never on co-tenants' data. This is what makes batched serving
     verifiable (a challenger reproduces stream A at the same B with any padding).
  3. THROUGHPUT  — agg_tok_s at B=1/2/4 (localhost ring = GPU-bound aggregate; the WAN-amortized number
     comes from the multi-box ring).

  python m25_batch_serve_driver.py
"""
import socket, os
import m25_stage as S
import m25_pipe as P
if os.environ.get("SHARD_TRANSPORT") != "libp2p":   # raw-wire mode: load SHARD_PSK (libp2p sidecar self-seals)
    import wire; wire.key_from_env()
from transformers import AutoTokenizer
from ngram_draft import NgramDrafter

HEAD = ("localhost", int(os.environ.get("HEAD_PORT", "29610")))
TAIL = ("localhost", int(os.environ.get("TAIL_PORT", "29611")))
tok = AutoTokenizer.from_pretrained(S.DIR, trust_remote_code=True)

# ONE persistent connection for all jobs (reset_batch clears stage state per job). Opening a fresh
# connection per job triggers serve's edge-close/reset, which breaks the head->tail nxt_sock — so reuse.
pipe = socket.create_connection(HEAD, timeout=600); pipe.setsockopt(*P.NODELAY)
ret = socket.create_connection(TAIL, timeout=600); ret.setsockopt(*P.NODELAY); ret.settimeout(600)
P.send_msg(ret, {"op": "hello_return"}); P.recv_msg(ret)

def run_batch(prompts, max_new=40, K=6):
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    drafters = [NgramDrafter(ng=3) for _ in prompts]
    return P.coordinate_pipe_batch(pipe, tok, msgs, K, max_new, 600, ret, drafters, prefill_chunk=512, max_ctx=131072)

# ---- 1. COHERENCE ----
print("=== 1. COHERENCE (B=3 distinct prompts) ===", flush=True)
P3 = ["Count from 1 to 30, separated by commas.",
      "List five European capital cities.",
      "Explain what gravity is in exactly one sentence."]
r = run_batch(P3)
for b, s in enumerate(r["streams"]):
    print(f"  stream {b}: {s['n_tokens']:3d}tok  {s['text'][:90]!r}", flush=True)
print(f"  agg_tok_s={r['agg_tok_s']:.2f} rounds={r['rounds']} dt={r['dt']:.2f}s", flush=True)

# ---- 2. DATA ISOLATION (the gate) ----
print("\n=== 2. DATA ISOLATION (stream A independent of batch-mates) ===", flush=True)
A = "Count from 1 to 30, separated by commas."
r1 = run_batch([A, "List five European capital cities.", "Explain gravity in one sentence."])
r2 = run_batch([A, "Write a haiku about the sea.", "What is 17 times 23?"])
a1, a2 = r1["streams"][0]["output_ids"], r2["streams"][0]["output_ids"]
iso = (a1 == a2)
print(f"  A with mates #1: {a1[:24]}", flush=True)
print(f"  A with mates #2: {a2[:24]}", flush=True)
print(f"  ISOLATION: {'PASS (identical -> data-isolated, verifiable)' if iso else 'FAIL (output depends on co-tenants!)'}", flush=True)
if not iso:
    n = min(len(a1), len(a2)); div = next((i for i in range(n) if a1[i] != a2[i]), n)
    print(f"  diverged at token {div} (len {len(a1)} vs {len(a2)})", flush=True)

# ---- 3. THROUGHPUT B=1/2/4 (localhost = GPU-bound aggregate) ----
print("\n=== 3. THROUGHPUT (localhost ring, GPU-bound aggregate) ===", flush=True)
base = None
for B in (1, 2, 4):
    pr = run_batch(["Count from 1 to 60, separated by commas."] * B, max_new=48)
    base = base or pr["agg_tok_s"]
    print(f"  B={B}: agg={pr['agg_tok_s']:6.2f} tok/s  rounds={pr['rounds']}  ({pr['agg_tok_s']/base:.2f}x vs B=1)", flush=True)
print("[driver] done", flush=True)
