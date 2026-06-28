"""Offline proof that M25_STATIC_KV (preallocated buffer + index_copy_) is BIT-IDENTICAL to the
grow-by-cat KV path in m25_stage.Layer.attn — across prefill chunks, spec-decode verify rounds, a
REJECT/rollback (re-draft from an earlier start_pos), and a reset+new job. m25_stage can't import
off-box, so we replicate the two KV-management bodies verbatim and compare the (kcur,vcur,total) each
step feeds into attention.

The read is :total exact in both modes, so once the KV slice matches, the downstream SDPA/naive attn
(already proven equivalent in m25_sdpa_test.py) is identical. Proves rollback correctness: a fixed-buffer
write at start_pos overwrites stale speculative KV exactly like cat's crop-to-start_pos + append.

  python research/m25_statickv_test.py
"""
import torch

NKV, HD = 8, 128
torch.manual_seed(0)


def cat_step():
    """Verbatim from the M25_STATIC_KV=0 branch: crop-to-start_pos then cat."""
    kc = vc = None
    def step(k, v, start):
        nonlocal kc, vc
        if kc is not None and kc.shape[2] > start:
            kc = kc[:, :, :start, :].contiguous(); vc = vc[:, :, :start, :].contiguous()
        if kc is None:
            kc, vc = k, v
        else:
            kc = torch.cat([kc, k], 2); vc = torch.cat([vc, v], 2)
        total = kc.shape[2]
        return kc, vc, total
    return step


def static_step(maxlen):
    """Verbatim from the M25_STATIC_KV=1 branch: index_copy_ at [start,start+s), read :total."""
    kc = torch.zeros(1, NKV, maxlen, HD)
    vc = torch.zeros(1, NKV, maxlen, HD)
    def step(k, v, start):
        s = k.shape[2]; total = start + s
        cp = torch.arange(start, total)
        kc.index_copy_(2, cp, k); vc.index_copy_(2, cp, v)
        return kc[:, :, :total, :], vc[:, :, :total, :], total
    return step
    # NOTE: reset() in static mode is a no-op (clen logical reset) — the next job's first write is at
    # start_pos=0 and reads are :total-bounded, so stale tail is never read. We model that by allocating
    # a fresh static_step() per job below (== buffer carried across reset; only :total ever read).


def kv(start, s, tag):
    """deterministic-but-distinct k/v for this (start,s,tag) so mis-writes are detectable."""
    g = torch.Generator().manual_seed(start * 1000 + s * 7 + hash(tag) % 97)
    return (torch.randn(1, NKV, s, HD, generator=g), torch.randn(1, NKV, s, HD, generator=g))


def run_sequence(ops):
    cat = cat_step(); sta = static_step(maxlen=4096)
    for i, (start, s, tag) in enumerate(ops):
        k, v = kv(start, s, tag)
        kc, vc, tc = cat(k, v, start)
        ks, vs, ts = sta(k, v, start)
        assert tc == ts == start + s, f"step {i} ({tag}): total mismatch cat={tc} static={ts}"
        assert torch.equal(kc, ks), f"step {i} ({tag}): K slice differs (cat vs static)"
        assert torch.equal(vc, vs), f"step {i} ({tag}): V slice differs"
        print(f"  step {i:2} {tag:14} start={start:4} s={s:3} total={tc:4}  K/V bit-identical")


def test_prefill_verify_rollback():
    # prefill chunks -> accepted verify rounds -> a REJECT (re-draft from an earlier start) -> continue
    # start_pos always tracks the committed length (continue) or rolls BACK below it (reject) — the
    # coordinator never skips forward past the current length, which is the static path's invariant.
    ops = [
        (0, 512, "prefill-0"), (512, 512, "prefill-1"), (1024, 300, "prefill-2"),   # 1324-token prompt
        (1324, 5, "verify-accept"),     # K+1=5 draft block, all accepted -> total 1329
        (1329, 5, "verify-accept2"),    # accepted -> 1334
        (1331, 5, "verify-REJECT"),     # rejection at offset 2: re-draft from 1331 (< prev total 1334): OVERWRITE stale
        (1336, 5, "verify-resume"),     # continue
    ]
    run_sequence(ops)
    print("[statickv] PASS 1 — static index_copy_ == cat crop+append across prefill/verify/ROLLBACK")


def test_reset_new_job():
    # a fresh job after reset: static buffer is reused (no zeroing); first write at 0 + :total read must
    # be byte-clean despite stale tail from the prior job.
    sta = static_step(maxlen=4096)
    # job A leaves the buffer populated to 1000
    for start, s in [(0, 600), (600, 400)]:
        k, v = kv(start, s, "A"); sta(k, v, start)
    # job B (== after reset): fresh cat vs the SAME reused static buffer, starting at 0
    cat = cat_step()
    for i, (start, s) in enumerate([(0, 50), (50, 5), (52, 5)]):
        k, v = kv(start, s, "B")
        kc, vc, tc = cat(k, v, start)
        ks, vs, ts = sta(k, v, start)
        assert torch.equal(kc, ks) and torch.equal(vc, vs), f"reset job step {i}: stale tail leaked into :total read"
    print("[statickv] PASS 2 — reset (reused buffer, no zeroing) is byte-clean (:total bounds the read)")


if __name__ == "__main__":
    test_prefill_verify_rollback()
    test_reset_new_job()
    print("\n[statickv] ALL PASS — M25_STATIC_KV is bit-identical to grow-by-cat (rollback + reset safe)")
