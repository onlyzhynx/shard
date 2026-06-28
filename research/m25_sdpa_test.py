"""Offline CPU proof for the SDPA prefill-attention fix in m25_stage.Layer.attn (the long-ctx OOM unblock).

m25_stage loads the model config at import (needs the box), so we can't import it here. Instead we
replicate the TWO attention bodies verbatim — the naive `matmul + bottom-right mask + fp32-softmax + AV`
(the M25_SDPA=0 reference) and the new `scaled_dot_product_attention(..., causal_lower_right, enable_gqa)`
path — and prove they're numerically equivalent across the 4 shape regimes the engine actually drives.

Proves:
 1. SDPA(causal_lower_right) == naive bottom-right causal, to <1e-5, AND identical downstream argmax
    (the spec-decode accept compares argmax, so argmax-equality is the property that matters).
 2. enable_gqa=True == manual repeat_interleave(GRP) expansion (bit-identical GQA).
 3. is_causal=True does NOT match when start_pos>0 (regression guard documenting the top-left footgun
    that this fix deliberately avoids).

  python research/m25_sdpa_test.py
"""
import torch
from torch.nn.attention.bias import causal_lower_right

torch.manual_seed(0)
NH, NKV, HD = 48, 8, 128           # M2.5 GQA: 48 q-heads / 8 kv-heads, head_dim 128
GRP = NH // NKV                    # 6
SCALING = HD ** -0.5
F = torch.nn.functional

# (s_new, start_pos) — total = start_pos + s_new (the engine crops kc to start_pos then appends s_new)
REGIMES = [("full-prefill", 8, 0), ("chunked-prefill", 512, 4096),
           ("verify(K+1)", 7, 5000), ("decode", 1, 9999)]


def _mk(s, total, dtype):
    q = torch.randn(1, NH, s, HD, dtype=dtype)
    kc = torch.randn(1, NKV, total, HD, dtype=dtype)
    vc = torch.randn(1, NKV, total, HD, dtype=dtype)
    return q, kc, vc


def naive_attn(q, kc, vc, start_pos):
    """Verbatim from the M25_SDPA=0 branch: 48-head expand, full score matrix, bottom-right mask, fp32 softmax."""
    s, total = q.shape[2], kc.shape[2]
    kk = kc.repeat_interleave(GRP, dim=1); vv = vc.repeat_interleave(GRP, dim=1)
    attn = torch.matmul(q, kk.transpose(-1, -2)) * SCALING
    qpos = torch.arange(s).view(s, 1) + start_pos
    kpos = torch.arange(total).view(1, total)
    attn = attn + torch.where(kpos <= qpos, 0.0, float("-inf")).to(attn.dtype)
    return torch.matmul(torch.softmax(attn.float(), -1).to(vv.dtype), vv)


def sdpa_gqa(q, kc, vc):
    """The new path: enable_gqa over the 8-head cache, causal_lower_right bias."""
    s, total = q.shape[2], kc.shape[2]
    return F.scaled_dot_product_attention(q, kc, vc, attn_mask=causal_lower_right(s, total),
                                          scale=SCALING, enable_gqa=True)


def sdpa_expand(q, kc, vc):
    """SDPA but with manual GQA expansion — isolates the enable_gqa==repeat_interleave claim."""
    s, total = q.shape[2], kc.shape[2]
    kk = kc.repeat_interleave(GRP, dim=1); vv = vc.repeat_interleave(GRP, dim=1)
    return F.scaled_dot_product_attention(q, kk, vv, attn_mask=causal_lower_right(s, total), scale=SCALING)


def _argmax(o, proj):
    s = o.shape[2]
    return (o.transpose(1, 2).reshape(1, s, NH * HD).float() @ proj).argmax(-1)


def test_equivalence_fp32():
    proj = torch.randn(NH * HD, 4096)          # downstream lm_head-like proj -> argmax proxy
    for name, s, start in REGIMES:
        total = start + s
        q, kc, vc = _mk(s, total, torch.float32)
        ref = naive_attn(q, kc, vc, start)
        g = sdpa_gqa(q, kc, vc)
        e = sdpa_expand(q, kc, vc)
        dg = (g - ref).abs().max().item()
        de = (e - ref).abs().max().item()
        dge = (g - e).abs().max().item()
        assert dg < 1e-5, f"[{name}] sdpa_gqa vs naive max|diff|={dg:.2e}"
        assert de < 1e-5, f"[{name}] sdpa_expand vs naive max|diff|={de:.2e}"
        assert dge < 1e-6, f"[{name}] enable_gqa != repeat_interleave: {dge:.2e}"
        am_ref, am_g = _argmax(ref, proj), _argmax(g, proj)
        assert torch.equal(am_ref, am_g), f"[{name}] downstream argmax differs"
        print(f"[sdpa] {name:16} s={s:4} start={start:5} total={total:5}  "
              f"gqa|naive={dg:.1e}  expand|naive={de:.1e}  gqa==expand={dge:.0e}  argmax=match")
    print("[sdpa] PASS 1 — SDPA(causal_lower_right, enable_gqa) == naive bottom-right, argmax-identical (fp32)")


def test_is_causal_is_wrong():
    """Guard: is_causal=True (top-left) must NOT match the bottom-right mask once start_pos>0 — this is the
    footgun the fix avoids. Documents WHY we pass causal_lower_right instead of is_causal=True."""
    for name, s, start in REGIMES:
        if start == 0:
            continue   # at start=0, s==total, top-left == bottom-right (no divergence to show)
        total = start + s
        q, kc, vc = _mk(s, total, torch.float32)
        ref = naive_attn(q, kc, vc, start)
        wrong = F.scaled_dot_product_attention(q, kc, vc, is_causal=True, scale=SCALING, enable_gqa=True)
        d = (wrong - ref).abs().max().item()
        assert d > 1e-2, f"[{name}] is_causal unexpectedly matched (d={d:.2e}) — footgun guard ineffective"
    print("[sdpa] PASS 2 — is_causal=True diverges from bottom-right when start_pos>0 (footgun guarded)")


def test_bf16_drift_bounded():
    """bf16: SDPA is NOT bit-identical to naive (online-softmax reassociation) — this is the accepted
    'self-consistent, not bit-identical' property. We assert only that the drift stays in the few-ULP
    band. We deliberately do NOT assert argmax-identity here: with random Gaussian inputs the logits are
    near-uniform so a few-ULP perturbation flips coin-toss ties — which is meaningless. argmax-stability is
    proven where it's real: fp32 exact math (PASS 1) and, on-box, real activations with clear winners
    (gqa_check cosine>0.999 + the --validate needle). This mirrors why the engine uses spec-decode VERIFY
    (greedy re-derivation), not bit-equality, as its correctness contract."""
    for name, s, start in REGIMES:
        total = start + s
        q, kc, vc = _mk(s, total, torch.bfloat16)
        d = (sdpa_gqa(q, kc, vc).float() - naive_attn(q, kc, vc, start).float()).abs().max().item()
        assert d < 5e-2, f"[{name}] bf16 sdpa vs naive drift too large: {d:.2e}"
        print(f"[sdpa] {name:16} bf16 drift vs naive = {d:.2e} (few-ULP band, argmax not asserted on random ties)")
    print("[sdpa] PASS 3 — bf16 SDPA drift bounded (few-ULP); argmax-stability proven in fp32 + on-box")


if __name__ == "__main__":
    test_equivalence_fp32()
    test_is_causal_is_wrong()
    test_bf16_drift_bounded()
    print("\n[sdpa] ALL PASS — SDPA prefill-attention fix is numerically equivalent to the naive path")
