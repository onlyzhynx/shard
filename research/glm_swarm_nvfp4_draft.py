"""GLM-5.2 NVFP4 swarm + DEEP draft speculative decoding — the real lever for 30-40 tok/s code.

A small tokenizer-compatible draft (GLM-4-9B-0414, same base vocab as GLM-5.2) drafts K tokens
autoregressively on the coord (local, fast) -> depth-K (vs the depth-1 MTP). The 744B GLM-5.2
chain verifies all K in ONE WAN traversal; accept the longest prefix the target agrees with
(greedy -> output token-identical to plain decode). Each accepted token saves a full WAN round-trip.

Stages: glm_swarm_nvfp4_kv.py (KV-cached, crops to start_pos -> rejected drafts roll back free).
Coord (this, pure-torch + a 9B HF model) holds the draft + GLM-5.2 embed/lm_head/norm.
  coord: python glm_swarm_nvfp4_draft.py coord --stage host:29600port --prompt "..." --max-new 128 --K 6
"""
import socket, time, argparse, torch
import glm_swarm_nvfp4_kv as KV
from glm_swarm_nvfp4_kv import dev, cfg, eps, send_msg, recv_msg
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

DRAFT = "/root/glm4_9b_draft"

def coord(stage_ep, prompt, max_new, K, ret_port=None):
    tok = AutoTokenizer.from_pretrained(KV.DIR, trust_remote_code=True)
    embed_w = KV.raw("model.embed_tokens.weight").to(torch.bfloat16).to(dev)
    lm_head_w = KV.raw("lm_head.weight").to(torch.bfloat16).to(dev)
    norm_w = KV.raw("model.norm.weight").float().to(dev)
    print("loading draft GLM-4-9B...", flush=True)
    draft = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
    print(f"draft loaded ({torch.cuda.memory_allocated()/1e9:.1f} GB)", flush=True)
    host, p = stage_ep.rsplit(":", 1)
    fwd = socket.create_connection((host, int(p)), timeout=300); fwd.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); fwd.settimeout(300)
    # RING (direct-return): we ship into the head only; the tail dials back to ret_port (7 hops, not 12).
    ret_srv = None; ret_conn = [None]
    if ret_port:
        ret_srv = socket.socket(); ret_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ret_srv.bind(("0.0.0.0", ret_port)); ret_srv.listen(1)
        print(f"coord(DRAFT RING K={K}) -> head {stage_ep}; tail returns on :{ret_port}", flush=True)
    else:
        print(f"coord(DRAFT K={K}, relay-back) -> {stage_ep}", flush=True)
    def chain(start, toks):
        h = torch.nn.functional.embedding(torch.tensor([toks], device=dev), embed_w)
        send_msg(fwd, start, h)
        if ret_port:
            if ret_conn[0] is None:                      # tail dials in once, on the first traversal
                ret_conn[0], _ = ret_srv.accept(); ret_conn[0].setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); ret_conn[0].settimeout(300)
            _, hb = recv_msg(ret_conn[0])
        else:
            _, hb = recv_msg(fwd)
        x = hb[0].float(); xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * norm_w
        return (xn.to(torch.bfloat16) @ lm_head_w.t()).float().argmax(-1).tolist()
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    eos = cfg.eos_token_id if isinstance(cfg.eos_token_id, list) else [cfg.eos_token_id]
    dcache = DynamicCache()
    t0 = time.time()
    r = chain(0, ids[0].tolist()); cur = r[-1]; pos = ids.shape[1]
    with torch.no_grad():
        draft(input_ids=ids, past_key_values=dcache, use_cache=True)        # prefill draft over prompt
    out = [cur]; rounds = 0; accepted = 0; dt_draft = 0.0; dt_verify = 0.0
    with torch.no_grad():
        while len(out) < max_new and cur not in eos:
            td = time.time()
            drafts = []; dtok = cur
            for i in range(K):
                dl = draft(input_ids=torch.tensor([[dtok]], device=dev), past_key_values=dcache, use_cache=True).logits
                dtok = int(dl[0, -1].argmax()); drafts.append(dtok)
            dt_draft += time.time() - td
            tv = time.time()
            r = chain(pos, [cur] + drafts)
            dt_verify += time.time() - tv
            n = 0
            for j in range(K):
                if drafts[j] == r[j]: n += 1
                else: break
            committed = drafts[:n] + [r[n]]
            out.extend(committed); cur = r[n]; pos += n + 1
            dcache.crop(pos)
            rounds += 1; accepted += n
            if any(t in eos for t in committed): break
    dt = time.time() - t0; ntok = len(out)
    full = ids[0].tolist() + out
    print(f"\nGENERATED {ntok} tokens in {dt:.1f}s = {ntok/dt:.2f} tok/s | {rounds} traversals | "
          f"mean accept {accepted/max(rounds,1):.2f} | {(accepted+rounds)/max(rounds,1):.2f} tok/traversal", flush=True)
    print(f"  time split: draft {dt_draft:.1f}s ({dt_draft/dt:.0%}) | verify {dt_verify:.1f}s ({dt_verify/dt:.0%})", flush=True)
    print("decoded:", repr(tok.decode(full, skip_special_tokens=True)[:600]), flush=True)
    return ntok / dt, (accepted + rounds) / max(rounds, 1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="role", required=True)
    p = sub.add_parser("coord"); p.add_argument("--stage", required=True)
    p.add_argument("--prompt", default="def quicksort(arr):"); p.add_argument("--max-new", type=int, default=128); p.add_argument("--K", type=int, default=6)
    p.add_argument("--ret-port", type=int, default=None)   # set -> RING direct-return (tail dials this)
    a = ap.parse_args(); coord(a.stage, a.prompt, a.max_new, a.K, a.ret_port)
