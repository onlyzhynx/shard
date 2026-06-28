"""Microbench (no model, pure torch, sm_120) of attention variants at the DECODE shape, to pick the
CUDA-graph masked-read approach BEFORE integrating. Per (total, bucket): correctness vs the flash baseline
+ EAGER and GRAPHED ms (graphed = launch overhead removed = real in-graph cost). Tests a short bucket and a
long one (the manual-matmul score-matrix penalty scales with bucket; flex shouldn't). Winner = correct +
graphed time closest to the flash baseline across buckets.
"""
import torch, time
F = torch.nn.functional
from torch.nn.attention.bias import causal_lower_right
dev = "cuda"
NH, NKV, HD = 48, 8, 128; GRP = NH // NKV; SC = HD ** -0.5
s = 9
torch.manual_seed(0)

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    flex_c = torch.compile(flex_attention, dynamic=False)
    HAVE_FLEX = True
except Exception as e:
    HAVE_FLEX = False; print("flex unavailable:", str(e)[:100])


def bench_eager(fn, n=80):
    for _ in range(8): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1000


def bench_graph(fn, n=80):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): g.replay()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1000, out


def run(total, alen):
    start = total - s
    q = (torch.randn(1, NH, s, HD, device=dev) * 0.3).bfloat16()
    kc = torch.zeros(1, NKV, alen, HD, device=dev, dtype=torch.bfloat16); kc[:, :, :total].normal_(0, 0.3)
    vc = torch.zeros(1, NKV, alen, HD, device=dev, dtype=torch.bfloat16); vc[:, :, :total].normal_(0, 0.3)
    qpos = (torch.arange(s, device=dev) + start).view(s, 1); kpos = torch.arange(alen, device=dev).view(1, alen)
    amask = torch.where(kpos <= qpos, 0.0, float("-inf")).to(torch.bfloat16)[None, None]

    def baseline(): return F.scaled_dot_product_attention(q, kc[:, :, :total], vc[:, :, :total], attn_mask=causal_lower_right(s, total), scale=SC, enable_gqa=True)
    def sdpa_add(): return F.scaled_dot_product_attention(q, kc, vc, attn_mask=amask, scale=SC, enable_gqa=True)
    def manual():
        kk = kc.repeat_interleave(GRP, 1); vv = vc.repeat_interleave(GRP, 1)
        a = torch.matmul(q, kk.transpose(-1, -2)) * SC + amask
        return torch.matmul(torch.softmax(a.float(), -1).to(vv.dtype), vv)
    variants = [("baseline flash :total", baseline), ("sdpa additive :alen", sdpa_add), ("manual :alen", manual)]
    if HAVE_FLEX:
        def _causal(b, h, qi, ki): return (start + qi) >= ki
        BM = create_block_mask(_causal, 1, 1, s, alen, device=dev)
        def flex(): return flex_c(q, kc, vc, block_mask=BM, enable_gqa=True, scale=SC)
        variants.append(("flex :alen", flex))

    ref = baseline()
    print(f"\n--- total={total} bucket={alen} ---")
    for name, fn in variants:
        try:
            e = (fn().float() - ref.float()).abs().max().item(); eg = bench_eager(fn)
            try:
                gms, gout = bench_graph(fn); ge = (gout.float() - ref.float()).abs().max().item()
                gstr = f"graph {gms:.3f}ms (err {ge:.0e})"
            except Exception as gx:
                gstr = f"graph FAILED: {str(gx).splitlines()[0][:54]}"
            print(f"  {name:22} err={e:.1e}  eager {eg:.3f}ms  {gstr}")
        except Exception as ex:
            print(f"  {name:22} FAILED: {str(ex).splitlines()[0][:64]}")


print(f"decode s={s} NH={NH}/{NKV} HD={HD}  flex={HAVE_FLEX}")
run(2000, 2048)        # short/medium context
run(15000, 16384)      # long context (manual score matrix is 8x bigger here)
