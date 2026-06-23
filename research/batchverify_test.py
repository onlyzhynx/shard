"""Single-box proof of the batched fast-verify (phase0/batchverify.py): B independent streams,
sitting at DIVERGENT committed lengths, verified in ONE fixed-shape CUDA-graph replay produce the
SAME committed tokens (and matching hidden states) as running each stream ALONE through the proven
batch=1 FastVerify. This de-risks DEPLOY_READINESS Â§2 (concurrent request batching): the crux was
that the fast-verify graph is batch=1, fixed-shape.

Loads ONE stage block of the real gpt-oss-120B (default the TAIL block, so we also get norm+lm_head
and can compare committed TOKENS, not just hidden states). Tests:
  A. one batched decode round at divergent starts -> per-stream output == solo FastVerify (the crux).
  B. T-round lockstep generation from a shared prefill -> per-round, per-stream tokens stay == solo.
  C. throughput: batched B-stream decode (one replay) vs B sequential solo replays -> aggregate tok/s.

Run on a box that already has the 120B:
  cd /root && SHARD_PSK=x python3 batchverify_test.py --stage 3 --nstages 4 --B 4 --K 4 \
      --prefixes 120,800,2000,50 --rounds 8 --max-ctx 8192 --dump /root/bv.json
"""
import argparse, json, time
import torch
from pipeline import load_stage
from fastverify import FastVerify
from batchverify import BatchedFastVerify


def snapshot(cache):
    return ([t.clone() for t in cache.k], [t.clone() for t in cache.v])


def restore(dst_cache, snap):
    for t, s in zip(dst_cache.k, snap[0]): t.copy_(s)
    for t, s in zip(dst_cache.v, snap[1]): t.copy_(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/gpt-oss-120b")
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--nstages", type=int, default=4)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--prefixes", default="120,800,2000,50", help="per-stream committed prefix lengths")
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lossless", action="store_true", help="per-stream MoE (bit-exact) instead of the fast batched MoE")
    ap.add_argument("--dump", default="")
    a = ap.parse_args()
    dev = "cuda"
    kp1 = a.K + 1
    prefixes = [int(x) for x in a.prefixes.split(",")][:a.B]
    assert len(prefixes) == a.B, "need one prefix length per stream"
    is_tail = a.stage == a.nstages - 1

    parts = load_stage(a.model, a.stage, a.nstages, device=dev)
    hidden = parts["_model"].config.hidden_size
    torch.manual_seed(a.seed)
    scale = 0.1                                      # rough activation scale; exact value irrelevant to the A/B

    # deterministic inputs: a prefix activation per stream + a chunk per (stream, round) + a per-stream
    # accept-advance pattern (1..kp1) so streams DIVERGE in committed length, the whole point.
    prefix_h = [torch.randn(1, L, hidden, dtype=torch.bfloat16, device=dev) * scale for L in prefixes]
    chunk_h = [[torch.randn(1, kp1, hidden, dtype=torch.bfloat16, device=dev) * scale
                for _ in range(a.rounds)] for _ in range(a.B)]
    advance = [[1 + ((b + t) % kp1) for t in range(a.rounds)] for b in range(a.B)]   # tokens committed/round

    def tail_tokens(h_out):                          # [1 or B, kp1, hidden] -> argmax token ids per position
        logits = parts["lm_head"](parts["norm"](h_out))
        return logits.argmax(-1)                      # [.., kp1]

    # ---- reference: each stream ALONE through the proven batch=1 FastVerify ----
    print(f"[ref] prefilling {a.B} solo streams (lengths {prefixes}) ...", flush=True)
    refs = [FastVerify(parts, maxlen=a.max_ctx, dev=dev) for _ in range(a.B)]
    starts0 = []
    prefix_snaps = []
    for b in range(a.B):
        refs[b].reset()
        refs[b].prefill(prefix_h[b], 0)
        starts0.append(prefixes[b])
        prefix_snaps.append(snapshot(refs[b].cache))          # committed prefix KV, before any decode round
    torch.cuda.synchronize()

    # ---- batched: seat each stream's prefill into row b, then decode all B together ----
    bv = BatchedFastVerify(parts, B=a.B, maxlen=a.max_ctx, dev=dev, lossless=a.lossless)
    print(f"[batched] mode={'LOSSLESS (per-stream MoE)' if a.lossless else 'FAST (batched MoE, FP-lossy)'}", flush=True)
    bv.reset()
    for b in range(a.B):
        bv.load_prefix(b, FakeCache(prefix_snaps[b]))

    # ===== Test A + B: lockstep rounds, per-stream divergent starts, compare every round =====
    ref_starts = list(starts0)
    bat_starts = list(starts0)
    tok_pos = tok_match = 0
    max_hdiff = 0.0
    near_ties = 0
    for t in range(a.rounds):
        # reference: one solo decode per stream at its own start
        ref_out = []
        ref_tok = []
        for b in range(a.B):
            o = refs[b].decode(chunk_h[b][t], ref_starts[b])    # [1, kp1, hidden]
            ref_out.append(o.clone())
            if is_tail: ref_tok.append(tail_tokens(o)[0].clone())
            ref_starts[b] += advance[b][t]
        # batched: one graph replay for all B
        hin = torch.cat([chunk_h[b][t] for b in range(a.B)], dim=0)   # [B, kp1, hidden]
        st = torch.tensor(bat_starts, device=dev)
        bo = bv.decode(hin, st)                                  # [B, kp1, hidden]
        if is_tail: bat_tok = tail_tokens(bo)
        for b in range(a.B):
            d = (bo[b].float() - ref_out[b][0].float()).abs().max().item()
            max_hdiff = max(max_hdiff, d)
            if is_tail:
                rt, btk = ref_tok[b], bat_tok[b]
                tok_pos += kp1
                eq = int((rt == btk).sum().item())
                tok_match += eq
                if eq < kp1:                                     # characterise any mismatch as an FP near-tie
                    logits = parts["lm_head"](parts["norm"](bo[b:b+1]))[0]
                    for j in range(kp1):
                        if rt[j] != btk[j]:
                            top2 = logits[j].topk(2).values
                            if (top2[0] - top2[1]).abs().item() < 5e-2: near_ties += 1
            bat_starts[b] += advance[b][t]
        torch.cuda.synchronize()
    tok_rate = (tok_match / tok_pos) if tok_pos else float("nan")
    print(f"\n[A/B] {a.rounds} rounds Ã— B={a.B} streams, divergent starts {starts0} (advance pattern varied):", flush=True)
    print(f"   max |hidden_batched - hidden_solo| = {max_hdiff:.4g}  (FP, same class as cross-K non-associativity)", flush=True)
    if is_tail:
        print(f"   committed-token agreement batched-vs-solo = {tok_match}/{tok_pos} = {tok_rate*100:.3f}%"
              f"   (mismatches at FP near-ties: {near_ties}/{tok_pos - tok_match if tok_pos>tok_match else 0})", flush=True)

    # ===== Test C: throughput, batched one-replay vs B sequential solo replays =====
    # warm + time the steady-state decode at a representative mid context.
    def time_solo(n_iter=30):
        for b in range(a.B): refs[b].decode(chunk_h[b][0], ref_starts[b])   # warm graphs
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(n_iter):
            for b in range(a.B): refs[b].decode(chunk_h[b][0], ref_starts[b])
        torch.cuda.synchronize(); return (time.time() - t0) / n_iter
    def time_batched(n_iter=30):
        st = torch.tensor(bat_starts, device=dev)
        for _ in range(3): bv.decode(torch.cat([chunk_h[b][0] for b in range(a.B)], 0), st)  # warm
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(n_iter):
            bv.decode(torch.cat([chunk_h[b][0] for b in range(a.B)], 0), st)
        torch.cuda.synchronize(); return (time.time() - t0) / n_iter
    solo_s = time_solo(); bat_s = time_batched()
    solo_tps = a.B * kp1 / solo_s
    bat_tps = a.B * kp1 / bat_s
    print(f"\n[C] block-verify throughput @ B={a.B}, kp1={kp1}:", flush=True)
    print(f"   B sequential solo replays : {solo_s*1e3:.2f} ms/round -> {solo_tps:.0f} tok/s aggregate", flush=True)
    print(f"   1 batched graph replay    : {bat_s*1e3:.2f} ms/round -> {bat_tps:.0f} tok/s aggregate", flush=True)
    print(f"   batched speedup           : {solo_s/bat_s:.2f}Ã— wall, {bat_tps/solo_tps:.2f}Ã— aggregate tok/s", flush=True)

    if a.dump:
        json.dump({"test": "batched-fast-verify", "model": a.model, "stage": a.stage, "nstages": a.nstages,
                   "B": a.B, "K": a.K, "rounds": a.rounds, "prefixes": prefixes, "is_tail": is_tail,
                   "max_hidden_diff": max_hdiff, "token_agreement": tok_rate, "token_match": tok_match,
                   "token_positions": tok_pos, "near_tie_mismatches": near_ties,
                   "solo_ms": solo_s*1e3, "batched_ms": bat_s*1e3,
                   "solo_tok_s": solo_tps, "batched_tok_s": bat_tps,
                   "speedup_wall": solo_s/bat_s, "speedup_tok_s": bat_tps/solo_tps}, open(a.dump, "w"))
        print(f"\n[dump] -> {a.dump}", flush=True)


class FakeCache:
    """thin adapter so BatchedFastVerify.load_prefix can read a (k_list, v_list) snapshot."""
    def __init__(self, snap): self.k, self.v = snap


if __name__ == "__main__":
    main()
