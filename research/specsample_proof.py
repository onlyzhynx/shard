"""Proof that phase0/specsample.py is LOSSLESS — the speculative-sampling acceptance produces
samples exactly from the target temperature/top-p distribution, for a deterministic drafter.

No GPU/model needed: it drives the REAL Sampler class (the same code the tail runs) on synthetic
logits and checks, over many draws, that the committed-token distribution equals the target
distribution p to within Monte-Carlo noise (total-variation distance -> 0). This isolates the
acceptance math from the model. The on-swarm `specpipe --sample-test` then confirms the same
property on the real gpt-oss-120B distribution.

  python3 research/specsample_proof.py
"""
import sys, os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase0"))
from specsample import Sampler


def tv(emp, p):
    """total-variation distance between an empirical histogram (counts) and a distribution p."""
    e = emp / emp.sum()
    return 0.5 * float((e - p).abs().sum())


def single_slot_dist(temp, top_p, top_k, draft_pick, V=32, N=120_000, seed=1234):
    """Draw N committed first-tokens from a 1-slot spec-sampling round with a DETERMINISTIC draft
    token, and compare to the target distribution p. The committed first token is r[0] (= d if the
    draft is accepted, else the residual correction), which must be distributed exactly as p.
    draft_pick selects which token the deterministic drafter proposes, relative to p."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(V, generator=g) * 2.0                  # one fixed conditioning context
    s = Sampler(temp=temp, top_p=top_p, top_k=top_k, seed=seed, device="cpu")
    p = s.dist(logits)                                           # the target distribution we must match
    order = torch.argsort(p, descending=True)
    if draft_pick == "argmax":     d = int(order[0])            # most-likely token (high accept rate)
    elif draft_pick == "median":   d = int(order[V // 2])       # mid-prob token
    elif draft_pick == "outside":  d = int(order[-1])           # least-likely / possibly filtered-out token
    else:                          d = int(draft_pick)
    two_row = torch.stack([logits, logits])                     # [K+1=2, V]: row0 predicts the token, row1 bonus
    emp = torch.zeros(V)
    for _ in range(N):
        r = s.accept(two_row, [d])                              # r[0] is the committed first token
        emp[r[0]] += 1
    d_tv = tv(emp, p)
    pd = float(p[d])
    return d_tv, pd, d


def main():
    torch.manual_seed(0)
    print("=== specsample.py losslessness proof (deterministic-drafter speculative sampling) ===\n")
    # N tuned so Monte-Carlo TV noise (~0.007 for V=32) sits well under the 0.02 gate.
    THRESH = 0.02
    configs = [
        ("temp=1.0  top_p=1.0  top_k=0 ", dict(temp=1.0, top_p=1.0,  top_k=0)),
        ("temp=0.7  top_p=1.0  top_k=0 ", dict(temp=0.7, top_p=1.0,  top_k=0)),
        ("temp=0.7  top_p=0.9  top_k=0 ", dict(temp=0.7, top_p=0.9,  top_k=0)),
        ("temp=1.0  top_p=0.95 top_k=0 ", dict(temp=1.0, top_p=0.95, top_k=0)),
        ("temp=0.8  top_p=1.0  top_k=10", dict(temp=0.8, top_p=1.0,  top_k=10)),
        ("temp=1.3  top_p=0.8  top_k=20", dict(temp=1.3, top_p=0.8,  top_k=20)),
    ]
    picks = ["argmax", "median", "outside"]
    worst = 0.0
    allok = True
    for label, cfg in configs:
        for pick in picks:
            d_tv, pd, d = single_slot_dist(draft_pick=pick, **cfg)
            ok = d_tv < THRESH
            allok &= ok
            worst = max(worst, d_tv)
            print(f"  {label} | draft={pick:7} (p(d)={pd:.3f}) -> TV(committed, target)={d_tv:.4f}  "
                  f"{'OK' if ok else 'FAIL'}")
        print()

    # greedy (temp=0) must be bit-identical to plain argmax
    s = Sampler(temp=0.0, device="cpu")
    g = torch.Generator().manual_seed(7)
    L = torch.randn(5, 64, generator=g)
    r = s.accept(L, [int(L[0].argmax()), int(L[1].argmax()), int(L[2].argmax()), int(L[3].argmax())])
    greedy_ok = r == L.argmax(-1).tolist()
    print(f"  greedy(temp=0) accept == argmax per position: {'OK' if greedy_ok else 'FAIL'}")
    allok &= greedy_ok

    print(f"\nworst TV across all sampled configs = {worst:.4f}  (gate {THRESH})")
    print("VERDICT:", "LOSSLESS — committed distribution == target distribution (within MC noise)"
          if allok else "FAIL — distribution mismatch")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
