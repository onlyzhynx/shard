"""GLM-5.2 NVFP4 swarm — PIPELINED speculative decoding (async draft overlap + async verify pipelining).

Builds on the ring (direct-return) transport: because every stage fire-forwards, multiple verify
chunks can be in flight at once. The coord drafts a continuous stream with GLM-4-9B and sends
overlapping K-token verify chunks WITHOUT waiting (`depth` in flight):
  - the draft for chunk j+1 runs while chunk j traverses the WAN          => async-draft overlap
  - chunks j+1..j+depth traverse the ring pipelined behind chunk j        => async-verify pipeline
Throughput approaches the ring's per-chunk THROUGHPUT, not its full latency, for each accept run. On
divergence the in-flight chunks are stale: dcache.crop rolls the draft back, the stages' crop-to-
start_pos rolls their KV back, and the coord discards the stale results. Consecutive chunks overlap by
1 position; the target "bonus" token is taken only on divergence (folded into the next chunk on full
accept) to keep positions aligned by K. Greedy acceptance => output byte-identical to plain decode.
Measured 16.6 tok/s (K=2 depth=6) on 6 scattered RTX PRO 6000, vs 2.94 sync ring / 1.99 relay-back.

NOTE: CUDA-graphing the draft (StaticCache + torch.compile reduce-overhead) is 3.8x faster in
isolation (49.7->13.1 ms/tok, see glm_draft_bench.py) but does NOT compose with speculative rollback:
StaticCache leaves rejected drafts in the buffer and even masking the stale tail + pinning position_ids
can't bit-match DynamicCache.crop, so g collapses 1.94->0.3 (see glm_draft_rollback.py). It's
cudagraph-speed XOR clean-rollback; DynamicCache.crop (here) keeps g and correctness.

  coord: python glm_swarm_nvfp4_pipe.py coord --stage head:port --ret-port 29600 --depth 6 --K 2 \
         --prompt "def quicksort(arr):" --max-new 96
"""
import socket, time, argparse, torch
import glm_swarm_nvfp4_kv as KV
from glm_swarm_nvfp4_kv import dev, cfg, eps, send_msg, recv_msg
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

DRAFT = "/root/glm4_9b_draft"

def coord(stage_ep, prompt, max_new, K, ret_port, depth):
    tok = AutoTokenizer.from_pretrained(KV.DIR, trust_remote_code=True)
    embed_w = KV.raw("model.embed_tokens.weight").to(torch.bfloat16).to(dev)
    lm_head_w = KV.raw("lm_head.weight").to(torch.bfloat16).to(dev)
    norm_w = KV.raw("model.norm.weight").float().to(dev)
    print("loading draft GLM-4-9B...", flush=True)
    draft = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
    print(f"draft loaded ({torch.cuda.memory_allocated()/1e9:.1f} GB)", flush=True)
    DVOCAB = draft.config.vocab_size           # 151552 < GLM-5.2's 154880: clamp target-only ids fed to the draft
    host, p = stage_ep.rsplit(":", 1)
    fwd = socket.create_connection((host, int(p)), timeout=300); fwd.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); fwd.settimeout(300)
    ret_srv = socket.socket(); ret_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ret_srv.bind(("0.0.0.0", ret_port)); ret_srv.listen(1); ret_conn = [None]
    print(f"coord(PIPE depth={depth} K={K}) -> head {stage_ep}; tail returns on :{ret_port}", flush=True)
    eos = cfg.eos_token_id if isinstance(cfg.eos_token_id, list) else [cfg.eos_token_id]

    def send_chunk(start, toks):
        send_msg(fwd, start, torch.nn.functional.embedding(torch.tensor([toks], device=dev), embed_w))
    def recv_logits():
        if ret_conn[0] is None:
            ret_conn[0], _ = ret_srv.accept(); ret_conn[0].setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); ret_conn[0].settimeout(300)
        _, hb = recv_msg(ret_conn[0])
        x = hb[0].float(); xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * norm_w
        return (xn.to(torch.bfloat16) @ lm_head_w.t()).float().argmax(-1).tolist()
    def dnext(t):                              # one draft step (DynamicCache appends); clamp out-of-draft-vocab ids
        inp = torch.tensor([[t if t < DVOCAB else 0]], device=dev)
        return int(draft(input_ids=inp, past_key_values=dcache, use_cache=True).logits[0, -1].argmax())

    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist(); L = len(ids)
    dcache = DynamicCache()
    with torch.no_grad():
        send_chunk(0, ids); r = recv_logits(); cur = r[-1]                 # prefill verify -> first token
        draft(input_ids=torch.tensor([[min(t, DVOCAB - 1) for t in ids]], device=dev), past_key_values=dcache, use_cache=True)
    out = [cur]; pos = L                       # cur = committed token at absolute position pos
    inflight = []; discard = 0                 # FIFO of (start_pos, drafts); stale results to skip
    send_pos = pos; tail_tok = cur             # next chunk's start_pos and its first (cur) token
    valid = 0; accepted = 0; wasted = 0; dt_draft = 0.0; dt_recv = 0.0
    def draft_k():
        nonlocal tail_tok
        ds = []; t = tail_tok
        for _ in range(K):
            t = dnext(t); ds.append(t)
        return ds
    t0 = time.time()
    with torch.no_grad():
        done = False
        while not done:
            while len(inflight) < depth and not done:                     # FILL pipeline
                _td = time.time(); ds = draft_k(); dt_draft += time.time() - _td
                send_chunk(send_pos, [tail_tok] + ds)
                inflight.append((send_pos, ds)); tail_tok = ds[-1]; send_pos += K
            _tr = time.time(); r = recv_logits(); dt_recv += time.time() - _tr
            sp, ds = inflight.pop(0)                                       # READ one result
            if discard > 0:                                               # stale (post-divergence) -> skip
                discard -= 1; wasted += 1; continue
            n = 0
            for j in range(K):
                if ds[j] == r[j]: n += 1
                else: break
            valid += 1; accepted += n
            if n == K:
                out.extend(ds); pos += K; cur = ds[-1]
            else:                                                          # divergence -> correct + flush
                out.extend(ds[:n] + [r[n]]); cur = r[n]; pos += n + 1
                discard = len(inflight)                                    # every chunk still in flight is stale
                dcache.crop(pos); tail_tok = cur; send_pos = pos          # crop draft cache back to the corrected prefix
            if len(out) >= max_new or cur in eos: done = True
    dt = time.time() - t0; ntok = len(out)
    if cur in eos and out and out[-1] in eos: out = out[:-1]
    print(f"\nGENERATED {ntok} tokens in {dt:.1f}s = {ntok/dt:.2f} tok/s | depth {depth} K {K} | "
          f"{valid} valid traversals (+{wasted} stale) | mean accept {accepted/max(valid,1):.2f} | "
          f"{(accepted+valid)/max(valid,1):.2f} tok/valid-traversal", flush=True)
    print(f"  time split: draft {dt_draft:.1f}s ({dt_draft/dt:.0%}) | recv-wait {dt_recv:.1f}s ({dt_recv/dt:.0%})", flush=True)
    print("decoded:", repr(tok.decode(ids + out, skip_special_tokens=True)[:600]), flush=True)
    return ntok / dt

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="role", required=True)
    p = sub.add_parser("coord"); p.add_argument("--stage", required=True)
    p.add_argument("--prompt", default="def quicksort(arr):"); p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--K", type=int, default=2); p.add_argument("--ret-port", type=int, default=29600)
    p.add_argument("--depth", type=int, default=6)
    a = ap.parse_args(); coord(a.stage, a.prompt, a.max_new, a.K, a.ret_port, a.depth)
