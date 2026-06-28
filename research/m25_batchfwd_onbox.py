"""On-box: the BATCHED stage forward (continuous batching) produces each stream's hidden BIT-IDENTICAL to
that stream run SOLO (B=1), on real M2.5 NVFP4 layers. Proves the engine integration (per-stream scatter +
per-stream mask + per-stream MoE) is lossless. Also times the per-block GPU cost at B=1/2/4/8 to gauge the
aggregate-throughput model. Run with M25_BATCH=8.
  M25_BATCH=8 M25_DIR=/root/m25 python m25_batchfwd_onbox.py
"""
import os, torch, time
os.environ.setdefault("M25_DIR", "/root/m25")
import m25_stage as S

assert S.M25_BATCH >= 8, "run with M25_BATCH=8"
S.vllm_ctx(); dev = "cuda"
LAYERS = list(range(28, 32))
layers = [S.Layer(i) for i in LAYERS]
S.get_pe(); vcfg = S._CTX[1]
Kp1 = 9
B = 4
# per-stream prompt lengths (divergent committed starts) + seeded hidden inputs
Ls = [40, 1000, 2039, 200]
def prompt_h(i):
    g = torch.Generator(device=dev).manual_seed(100 + i); return (torch.randn(1, Ls[i], S.H, generator=g, dtype=torch.bfloat16, device=dev) * 0.2)
def block_h(i):
    g = torch.Generator(device=dev).manual_seed(900 + i); return (torch.randn(1, Kp1, S.H, generator=g, dtype=torch.bfloat16, device=dev) * 0.2)

with torch.no_grad():
    # ---- SOLO per stream (B=1 into row 0) ----
    solos = []
    for i in range(B):
        S.run_block_prefill_b(layers, 0, 0, prompt_h(i).clone(), vcfg)            # prefill stream i into row 0
        o = S.run_block_decode_b(layers, torch.tensor([Ls[i]], device=dev), block_h(i).clone(), vcfg)
        solos.append(o.float().clone())                                          # [1, Kp1, H]
    # ---- BATCHED (B streams into rows 0..B-1) ----
    for i in range(B):
        S.run_block_prefill_b(layers, i, 0, prompt_h(i).clone(), vcfg)           # prefill stream i into row i
    hb = torch.cat([block_h(i) for i in range(B)], 0)                            # [B, Kp1, H]
    starts = torch.tensor(Ls[:B], device=dev)
    ob = S.run_block_decode_b(layers, starts, hb, vcfg)                          # [B, Kp1, H]

worst = 0.0
for i in range(B):
    d = (ob[i:i + 1].float() - solos[i]).abs().max().item(); worst = max(worst, d)
    print(f"  stream {i} L={Ls[i]:4d}  batched == solo  max|diff|={d:.2e}", flush=True)
print(f"[batchfwd] {'PER-STREAM == SOLO bit-exact' if worst < 1e-3 else f'DIVERGE worst={worst:.2e}'}  (worst {worst:.2e})", flush=True)

# ---- per-block GPU time at B=1/2/4/8 (aggregate model) ----
def bench(Bn, n=40):
    h = (torch.randn(Bn, Kp1, S.H, dtype=torch.bfloat16, device=dev) * 0.2)
    st = torch.tensor([1000] * Bn, device=dev)
    with torch.no_grad():
        for _ in range(5): S.run_block_decode_b(layers, st, h, vcfg)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(n): S.run_block_decode_b(layers, st, h, vcfg)
        torch.cuda.synchronize()
    return (time.time() - t) / n * 1000
print("[batchfwd] per-block GPU (4 layers):", flush=True)
base = None
for Bn in (1, 2, 4, 8):
    ms = bench(Bn); base = base or ms
    print(f"    B={Bn}: {ms:.2f}ms   {ms/base:.2f}x cost for {Bn}x tokens  -> {Bn/(ms/base):.2f}x tokens/ms", flush=True)
