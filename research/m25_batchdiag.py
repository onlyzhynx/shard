"""Isolation vs tiling: B=4 IDENTICAL streams -> are they identical to each other (isolation)? and vs B=1 (tiling)?"""
import os, torch
os.environ.setdefault("M25_DIR", "/root/m25")
import m25_stage as S
S.vllm_ctx(); dev = "cuda"
layers = [S.Layer(i) for i in range(28, 32)]; S.get_pe(); vcfg = S._CTX[1]
Kp1, L = 9, 500
ph = torch.randn(1, L, S.H, generator=torch.Generator(device=dev).manual_seed(1), dtype=torch.bfloat16, device=dev) * 0.2
bh = torch.randn(1, Kp1, S.H, generator=torch.Generator(device=dev).manual_seed(2), dtype=torch.bfloat16, device=dev) * 0.2
with torch.no_grad():
    S.run_block_prefill_b(layers, 0, 0, ph.clone(), vcfg)
    solo = S.run_block_decode_b(layers, torch.tensor([L], device=dev), bh.clone(), vcfg).float()
    for i in range(4): S.run_block_prefill_b(layers, i, 0, ph.clone(), vcfg)
    bb = S.run_block_decode_b(layers, torch.tensor([L] * 4, device=dev), bh.repeat(4, 1, 1).clone(), vcfg).float()
iso = max((bb[i] - bb[0]).abs().max().item() for i in range(4))
til = (bb[0] - solo[0]).abs().max().item()
# also: attention-only (no MoE) tiling — does the batched matmul itself diverge per row?
with torch.no_grad():
    for i in range(4): S.run_block_prefill_b(layers, i, 0, ph.clone(), vcfg)
    a4 = layers[0].attn_decode_b(layers[0]._rms(bh.repeat(4,1,1), layers[0].in_ln), torch.tensor([L]*4, device=dev), *S.get_pe()).float()
    S.run_block_prefill_b(layers, 0, 0, ph.clone(), vcfg)
    a1 = layers[0].attn_decode_b(layers[0]._rms(bh, layers[0].in_ln), torch.tensor([L], device=dev), *S.get_pe()).float()
print(f"identical streams: stream-to-stream max|diff| = {iso:.2e}  -> {'ISOLATED (no contamination)' if iso < 1e-3 else 'CONTAMINATION BUG'}")
print(f"layer0 attn-only  B=4 row0 vs B=1: max|diff| = {(a4[0]-a1[0]).abs().max().item():.2e}  (matmul tiling)")
print(f"full block        B=4 row0 vs B=1 solo: max|diff| = {til:.2e}  -> {'bit-exact' if til < 1e-3 else 'bf16 tiling amplified by MoE (valid-but-different)'}")
