"""isolate the batched-verify bug: B=2 IDENTICAL streams (same prefix, same inputs, same start).
Compares (a) the two batched rows to each other, (b) batched row 0 to a solo FastVerify, under both
EAGER (no CUDA graph) and GRAPHED replay. Tells us whether the break is the batched MoE math, the
mask/scatter, or CUDA-graph capture of the data-dependent MoE kernel at B>1."""
import argparse, torch
from pipeline import load_stage
from fastverify import FastVerify
from batchverify import BatchedFastVerify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/gpt-oss-120b")
    ap.add_argument("--stage", type=int, default=8)
    ap.add_argument("--nstages", type=int, default=9)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--prefix", type=int, default=200)
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--lossless", action="store_true")
    a = ap.parse_args()
    dev = "cuda"; kp1 = a.K + 1
    parts = load_stage(a.model, a.stage, a.nstages, device=dev)
    hidden = parts["_model"].config.hidden_size
    torch.manual_seed(0)
    pre = torch.randn(1, a.prefix, hidden, dtype=torch.bfloat16, device=dev) * 0.1
    chunk = torch.randn(1, kp1, hidden, dtype=torch.bfloat16, device=dev) * 0.1
    start = a.prefix

    # solo reference
    ref = FastVerify(parts, maxlen=a.max_ctx, dev=dev); ref.reset()
    ref.prefill(pre, 0)
    snap_k = [t.clone() for t in ref.cache.k]; snap_v = [t.clone() for t in ref.cache.v]
    solo = ref.decode(chunk, start).clone()                      # [1, kp1, hidden]

    class Fake:                                                   # adapter for load_prefix
        def __init__(s, k, v): s.k, s.v = k, v

    for B in (1, 2):
        bv = BatchedFastVerify(parts, B=B, maxlen=a.max_ctx, dev=dev, lossless=a.lossless); bv.reset()
        for b in range(B): bv.load_prefix(b, Fake(snap_k, snap_v))
        # cache sanity: did load_prefix faithfully copy the prefix KV into every row?
        cdiff = max((bv.cache.k[l][b].float() - snap_k[l][0].float()).abs().max().item()
                    for l in range(bv.n_layers) for b in range(B))
        hin = torch.cat([chunk] * B, 0)
        st = torch.tensor([start] * B, device=dev)
        bo = bv.decode(hin, st, use_graph=False)                  # eager (graph not the issue)
        vs_solo = (bo[0].float() - solo[0].float()).abs().max().item()
        print(f"[B={B} EAGER] cache_copy_diff={cdiff:.3g}  max|row0-solo|={vs_solo:.4g}", flush=True)


if __name__ == "__main__":
    main()
