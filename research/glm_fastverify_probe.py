"""Is the per-stage NVFP4 verify compute launch-bound (CUDA-graph-able, like gpt-oss fast-verify)?
Time a stage block (a few nvfp4 layers) eager for 1 vs 7 tokens, then CUDA-graph it and compare.
Big eager time that scales with tokens + a big graph speedup => fast-verify is the lever to make
the WAN verify RTT-bound. run on the coord (has layers 6-9). python glm_fastverify_probe.py"""
import time, torch
import glm_swarm_nvfp4_kv as KV
from glm_swarm_nvfp4_kv import H, dev, Layer, _get_pe, run_block
KV._vllm_ctx()
vcfg = KV._VC
LIDS = [6, 7, 8, 9]
layers = [Layer(i) for i in LIDS]
pe = _get_pe()
print(f"{len(layers)} layers loaded ({torch.cuda.memory_allocated()/1e9:.1f} GB)", flush=True)

def bench(ntok, iters=20):
    for L in layers: L.reset()
    h = torch.randn(1, ntok, H, dtype=torch.bfloat16, device=dev) * 0.1
    # warm
    for _ in range(3):
        for L in layers: L.reset()
        run_block(layers, 0, h, vcfg)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters):
        for L in layers: L.reset()
        run_block(layers, 0, h, vcfg)
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000  # ms

e1 = bench(1); e7 = bench(7)
nl = len(layers)
print(f"\nEAGER per-{nl}-layer block:  1 tok = {e1:.1f} ms   7 tok = {e7:.1f} ms", flush=True)
print(f"  -> per LAYER: 1tok {e1/nl:.2f} ms, 7tok {e7/nl:.2f} ms | per-token slope {(e7-e1)/6/nl:.2f} ms/layer/tok", flush=True)
# extrapolate to the full 78-layer model (the verify compute, one direction)
scale = 78 / nl
print(f"  -> full 78-layer verify COMPUTE (1 dir): 1tok ~{e1*scale:.0f} ms, 7tok ~{e7*scale:.0f} ms", flush=True)
print("If 7-tok >> 1-tok and both are 10s-of-ms/layer -> launch-bound -> CUDA-graph fast-verify wins big.", flush=True)
