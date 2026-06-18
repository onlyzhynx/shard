"""Nail the verify WAN/compute split on the LIVE chain. Time chain(fixed_pos, ntok) for a range of
ntok at a FIXED start_pos (stages crop to start_pos -> idempotent, clean repeats). Slope vs ntok =
on-the-wire compute/token (should be ~0 if memory-bound, per the probe); intercept = WAN loop + fixed.
A huge intercept + flat slope => the WAN relay-back is the bottleneck => direct-return is the lever,
NOT fast-verify.  coord: python glm_wan_probe.py --stage host:port"""
import socket, time, argparse, torch
import glm_swarm_nvfp4_kv as KV
from glm_swarm_nvfp4_kv import dev, send_msg, recv_msg

def main(stage_ep):
    embed_w = KV.raw("model.embed_tokens.weight").to(torch.bfloat16).to(dev)
    host, p = stage_ep.rsplit(":", 1)
    s = socket.create_connection((host, int(p)), timeout=300); s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); s.settimeout(300)
    def chain(start, toks):
        h = torch.nn.functional.embedding(torch.tensor([toks], device=dev), embed_w)
        t = time.time(); send_msg(s, start, h); recv_msg(s); return (time.time() - t) * 1000  # ms round-trip
    POS = 64
    for _ in range(3): chain(POS, [100])           # warm the path
    print(f"live chain round-trip (start_pos={POS}, full 6-stage relay-back loop):", flush=True)
    for ntok in [1, 2, 4, 7, 10, 16]:
        ms = sorted(chain(POS, [100 + j for j in range(ntok)]) for _ in range(6))[1]  # 2nd-best (drop tail)
        print(f"  {ntok:2d} tok  ->  {ms:7.1f} ms   ({ms/ntok:.1f} ms/tok)", flush=True)
    print("\nFlat across ntok => compute negligible on the wire; the ms IS the WAN loop.", flush=True)
    print("That number is what direct-return (kill the 6-hop relay-back) roughly halves.", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True)
    main(ap.parse_args().stage)
