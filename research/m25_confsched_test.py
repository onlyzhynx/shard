"""Offline proof for the confidence-scheduled depth port in m25_pipe.coordinate_pipe.

NO GPU / model / network. Two properties:

 1. LOSSLESS PORT — with M25_CONF_SCHED off vs on, coordinate_pipe returns BYTE-IDENTICAL
    output_ids (== the deterministic greedy 'truth' stream) across BOTH a high-acceptance
    and a zero-acceptance drafter. Confidence scheduling only changes how many speculative
    chunks are in flight; the verify/accept path is untouched, so the committed output cannot
    move. Proven over a REAL in-memory ring driving the ACTUAL coordinate_pipe bookkeeping
    (heavy on-box deps stubbed exactly like research/m25_sweep_test.py).

 2. THROTTLE — the ConfidenceScheduler, driven the way the loop drives it (value() at fill,
    observe(n,K) after accept), stays at full depth on high acceptance (=> identical to a
    fixed-depth ring on copy/retrieval, no regression to the warm baseline) and throttles
    toward depth 1 on a sustained low-acceptance streak (the win), then reopens on recovery.

  python research/m25_confsched_test.py
"""
import sys, os, types, threading, queue

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase0"))   # so `import m25_pipe` + the real confidence.py resolve

PROMPT_IDS = list(range(2, 22))     # 20-token prompt (< prefill_chunk -> single-verify prefill path)
EOS = 999999                        # sentinel that never appears in the truth stream -> no early stop


def truth(p):
    """Deterministic greedy 'target' token at absolute position p (never EOS). A position-keyed
    oracle is a faithful greedy target for spec-decode: only matched-or-corrected tokens are ever
    committed, so the committed stream equals truth regardless of draft quality (= losslessness)."""
    return (p * 2654435761) % 5000


# ---- stub heavy on-box deps so `import m25_pipe` works anywhere (same approach as m25_sweep_test) ----
def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


class _Chan:
    """A queue masquerading as a socket: send_msg/recv_msg below just put/get on it."""
    def __init__(self): self.q = queue.Queue()
    def settimeout(self, t): pass


_stub("m25_stage", H=3072, DIR="/tmp/none", EPS=1e-6, raw=lambda *a, **k: None,
      vllm_ctx=lambda *a, **k: None, Layer=object, run_block=lambda *a, **k: None, _CTX=(None, None))
_stub("m25_tools", render_ids=lambda tok, messages, tools=None: list(PROMPT_IDS),
      parse_completion=lambda t: {"content": t, "reasoning_content": "", "tool_calls": []})
_stub("node_kv", send_msg=lambda sock, obj: sock.q.put(obj),
      recv_msg=lambda sock: sock.q.get(timeout=15), EDGE_ERRORS=(Exception,), TransportError=Exception)
_stub("receipt", ReceiptSigner=None, load_or_make_node_key=lambda *a, **k: None,
      verify_receipt=lambda *a, **k: None, verify_coverage=lambda *a, **k: None)

import m25_pipe   # noqa: E402  (after stubs)


class _FakeTok:
    eos_token_id = EOS
    def decode(self, ids, skip_special_tokens=True): return ",".join(map(str, ids))


class _TruthDrafter:
    """Proposes the correct next tokens -> ~100% acceptance -> scheduler stays at full depth."""
    def request(self, ids, k): self._ids = list(ids); self._k = k
    def fetch(self): b = len(self._ids); return [truth(b + i) for i in range(self._k)]


class _WrongDrafter:
    """Always wrong -> 0% acceptance -> scheduler throttles hard. Output must STILL be lossless."""
    def request(self, ids, k): self._ids = list(ids); self._k = k
    def fetch(self): b = len(self._ids); return [(truth(b + i) + 1) % 5000 for i in range(self._k)]


def _ring(pipe_in, ret_out, stop):
    """Deterministic greedy tail: for a verify of L token_ids at `start`, return the argmax at each
    position = the predicted token at absolute pos (start+j+1). reset->ack, receipt->[]."""
    while not stop.is_set():
        try:
            msg = pipe_in.q.get(timeout=0.25)
        except queue.Empty:
            continue
        op = msg.get("op")
        if op == "reset":
            ret_out.q.put("ok")
        elif op == "receipt":
            ret_out.q.put([])
        else:  # verify
            start = msg["start"]; n = len(msg["token_ids"])
            ret_out.q.put([truth(start + j + 1) for j in range(n)])


def _run(drafter, conf_on, K=8, depth=6, max_new=40):
    if conf_on:
        os.environ["M25_CONF_SCHED"] = "1"
    else:
        os.environ.pop("M25_CONF_SCHED", None)
    pipe = _Chan(); ret = _Chan(); stop = threading.Event()
    t = threading.Thread(target=_ring, args=(pipe, ret, stop), daemon=True); t.start()
    try:
        res = m25_pipe.coordinate_pipe(
            pipe_sock=pipe, tok=_FakeTok(), messages=[{"role": "user", "content": "x"}],
            K=K, max_new=max_new, timeout=15, depth=depth, ret_sock=ret, local_draft=drafter,
            tools=None, prefill_chunk=0, max_ctx=0)
    finally:
        stop.set(); t.join(timeout=2)
    return res


def test_lossless_on_vs_off():
    """ON and OFF must agree byte-for-byte AND equal the truth stream, for both drafters."""
    for name, mk in [("high-accept", _TruthDrafter), ("zero-accept", _WrongDrafter)]:
        off = _run(mk(), conf_on=False)["output_ids"]
        on = _run(mk(), conf_on=True)["output_ids"]
        expect = [truth(len(PROMPT_IDS) + i) for i in range(len(on))]
        assert off == on, f"[{name}] conf ON != OFF: {off[:8]} vs {on[:8]}"
        assert on == expect, f"[{name}] output != greedy truth stream"
        assert len(on) >= 40, f"[{name}] expected >= max_new tokens, got {len(on)}"
    print("[confsched] PASS 1 — output byte-identical ON vs OFF == greedy truth (high + zero accept)")


def test_high_accept_does_not_change_depth():
    """On a high-acceptance task the scheduled depth must sit at the ceiling every round, so the
    confidence path is a no-op vs fixed depth (no regression to the warm copy/retrieval baseline)."""
    res_on = _run(_TruthDrafter(), conf_on=True)
    assert res_on["final_confidence"] is not None and res_on["final_confidence"] > 0.9, \
        f"high-accept EMA should be ~1.0, got {res_on['final_confidence']}"
    res_off = _run(_TruthDrafter(), conf_on=False)
    assert res_off["final_confidence"] is None, "OFF must not allocate a scheduler"
    print(f"[confsched] PASS 2 — high accept keeps full depth (EMA={res_on['final_confidence']:.3f}); OFF is inert")


def test_throttle_curve():
    """Drive the scheduler exactly as coordinate_pipe does (value() at fill, observe(n,K) after) and
    assert: high accept -> full depth; sustained zero -> throttle to 1; recovery -> reopen."""
    from confidence import ConfidenceScheduler
    K, depth = 8, 6

    def trace(accept_seq):
        c = ConfidenceScheduler(1, depth, lo=0.3, hi=0.7); out = []
        for nacc in accept_seq:
            out.append(c.value())   # depth used THIS round (read before observe, as the loop does)
            c.observe(nacc, K)
        return out, c

    hi, _ = trace([K] * 6)
    assert all(v == depth for v in hi), f"high accept must stay full depth, got {hi}"

    lo, c_lo = trace([0] * 8)
    assert lo[0] == depth and lo[-1] == 1, f"sustained zero accept must throttle to 1, got {lo}"

    # recover: prime low, then feed perfect acceptance -> reopen to full depth
    for _ in range(8): c_lo.observe(0, K)
    for _ in range(8): c_lo.observe(K, K)
    assert c_lo.value() == depth, f"recovery should reopen to {depth}, got {c_lo.value()}"
    print(f"[confsched] PASS 3 — throttle curve: full={hi[0]}, floor={lo[-1]}, recovered={c_lo.value()}")


if __name__ == "__main__":
    from confidence import _selftest
    _selftest()                       # the module's own logic gate
    test_lossless_on_vs_off()
    test_high_accept_does_not_change_depth()
    test_throttle_curve()
    print("\n[confsched] ALL PASS — confidence-scheduled depth ported to m25_pipe, lossless + opt-in")
