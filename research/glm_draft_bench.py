"""Isolate + speed up the GLM-4-9B draft (the 94% bottleneck of the pipelined swarm). No swarm needed.
Measures: (1) actual StaticCache buffer size, (2) eager ms/token, (3) torch.compile(reduce-overhead,
CUDA-graph) ms/token with static input buffers, (4) the GLM-5.2(154880) vs GLM-4-9B(151552) vocab gap.
  python glm_draft_bench.py [--compile] [--n 128]"""
import time, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

DRAFT = "/root/glm4_9b_draft"; dev = "cuda"

def main(do_compile, n):
    tok = AutoTokenizer.from_pretrained(DRAFT, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
    DVOCAB = m.config.vocab_size
    print(f"draft loaded ({torch.cuda.memory_allocated()/1e9:.1f} GB), vocab={DVOCAB}", flush=True)
    ids = tok("def quicksort(arr):", return_tensors="pt").input_ids.to(dev); L = ids.shape[1]
    MAXLEN = L + n + 16
    cache = StaticCache(config=m.config, max_cache_len=MAXLEN, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        m(input_ids=ids, past_key_values=cache, cache_position=torch.arange(L, device=dev), use_cache=True)
    kb = cache.layers[0].keys
    print(f"requested MAXLEN={MAXLEN} | actual key buffer = {tuple(kb.shape) if kb is not None else None}", flush=True)

    # static input buffers so reduce-overhead's CUDA graph sees stable addresses
    inp = torch.zeros((1, 1), dtype=torch.long, device=dev)
    cpos = torch.zeros((1,), dtype=torch.long, device=dev)
    step = torch.compile(m, mode="reduce-overhead", fullgraph=False) if do_compile else m
    def dstep(t, p):
        inp[0, 0] = t; cpos[0] = p
        return int(step(input_ids=inp, past_key_values=cache, cache_position=cpos, use_cache=True).logits[0, -1].argmax())

    cur = int(m(input_ids=ids, past_key_values=cache, cache_position=torch.arange(L, device=dev), use_cache=True).logits[0, -1].argmax())
    with torch.no_grad():
        if do_compile:
            for w in range(8): dstep(cur, L + w)        # warm the graph
        torch.cuda.synchronize(); t0 = time.time(); t = cur; p = L
        for _ in range(n):
            t = dstep(t, p); p += 1
        torch.cuda.synchronize(); dt = time.time() - t0
    print(f"{'COMPILED' if do_compile else 'EAGER'} decode: {n} tok in {dt*1000:.0f} ms = {dt/n*1000:.1f} ms/tok = {n/dt:.0f} tok/s", flush=True)

    # vocab-gap test: feed a GLM-5.2-only token id (>= draft vocab) — does it crash?
    try:
        with torch.no_grad(): dstep(min(154000, DVOCAB + 100), L)
        print("fed out-of-vocab token: OK (no crash)", flush=True)
    except Exception as e:
        print(f"fed out-of-vocab token: CRASH -> {type(e).__name__} (confirms vocab gap; must clamp)", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--compile", action="store_true"); ap.add_argument("--n", type=int, default=128)
    a = ap.parse_args(); main(a.compile, a.n)
