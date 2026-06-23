"""Lossless speculative SAMPLING for the distributed verify path (temperature / top-p / top-k).

The fast-verify ring is greedy-only today: the tail returns argmax per position and the
coordinator commits the longest draft prefix that matches. That gives ONE valid greedy decode,
but real workloads want temperature/top-p sampling — and naive "sample at the tail instead of
argmax" is NOT lossless (it changes the accept criterion and the committed distribution drifts
from the target's true sampling distribution).

This module makes spec-decode produce samples EXACTLY from the target's temperature/top-p
distribution, using the speculative-sampling rejection rule specialised to a DETERMINISTIC
drafter (the n-gram / prompt-lookup drafter proposes one token per slot, so its proposal
distribution q is a point mass at the drafted token):

    accept the drafted token d_j with probability  min(1, p_j(d_j)/q(d_j)) = p_j(d_j)
    on the first rejection at slot m, sample the correction from the residual
        norm( (p_m - q)_+ ) = norm( p_m with d_m removed )
    if all K slots accept, sample one bonus token from p_K (the free target sample).

Proof it is lossless (q a point mass at d):  P(out=d) = P(accept) = p(d);  for y!=d,
P(out=y) = P(reject)*residual(y) = (1-p(d)) * p(y)/(1-p(d)) = p(y).  So out ~ p exactly,
for every slot — the committed sequence is distributed identically to plain autoregressive
sampling from the target at the same temperature/top-p. (Leviathan'22 / Chen'22, deterministic-q
case.) The local proof in research/specsample_proof.py checks this to TV ~ 0 numerically; the
on-swarm test (`specpipe --sample-test`) confirms it on the real gpt-oss-120B distribution.

The clever part for the wire: the tail returns a DOCTORED result vector r (length K+1) where
r[j]=d_j for accepted slots, r[m]=correction at the first rejection (always != d_m, since the
residual zeroes d_m), and r[K]=bonus when all accept. Feeding r into the coordinator's existing
equality accept-loop (`n = longest j with d_j==r[j]; commit d[:n]+[r[n]]`) reproduces exactly the
speculative-sampling result — so the coordinator's decode loop is UNCHANGED and the WAN payload
stays a tiny int list. temp<=0 falls back to argmax → bit-identical to the current greedy path.
"""
import torch


class Sampler:
    """Temperature / top-k / top-p sampler with a seeded generator (reproducible receipts).

    temp<=0 means greedy (argmax) — the tail then behaves bit-identically to the legacy path.
    dist() returns the filtered, renormalised next-token distribution p over the full vocab; the
    speculative-sampling acceptance and the residual/bonus draws all read from this same p, so the
    proposal is judged against precisely the distribution we sample from."""

    def __init__(self, temp=0.0, top_p=1.0, top_k=0, seed=0, device="cuda"):
        self.temp = float(temp)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.greedy = self.temp <= 0.0
        self.device = device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)

    def dist(self, logits_row):
        """logits_row: [V] (any dtype/device) -> [V] float32 probability vector on self.device.
        temperature, then top-k, then top-p (nucleus) filtering, renormalised over the kept set."""
        z = logits_row.to(self.device, torch.float32) / max(self.temp, 1e-6)
        if self.top_k and 0 < self.top_k < z.numel():
            kth = torch.topk(z, self.top_k).values[-1]
            z = torch.where(z < kth, z.new_full((), float("-inf")), z)
        p = torch.softmax(z, dim=-1)
        if 0.0 < self.top_p < 1.0:
            sp, idx = torch.sort(p, descending=True)
            cum = torch.cumsum(sp, dim=-1)
            keep = cum - sp <= self.top_p                 # keep through the token that crosses top_p
            sp = torch.where(keep, sp, sp.new_zeros(()))
            p = torch.zeros_like(p).scatter_(0, idx, sp)
            p = p / p.sum()
        return p

    def sample(self, probs):
        """draw one token id from a probability vector (already normalised)."""
        return int(torch.multinomial(probs, 1, generator=self.gen).item())

    def uniform(self):
        return float(torch.rand((), generator=self.gen, device=self.device).item())

    # ---- the two ops the tail calls -------------------------------------------------
    def sample_logits(self, logits_row):
        """one token from a single logits row — greedy => argmax, else temp/top-p sample.
        used for the prefill's last position (the first generated token) and the bonus token."""
        if self.greedy:
            return int(logits_row.argmax(-1).item())
        return self.sample(self.dist(logits_row))

    def accept(self, logits, draft):
        """speculative-sampling acceptance for a deterministic draft.
        logits: [K+1, V] (row j predicts the token after chunk slot j); draft: list of K ids.
        returns a doctored result list r (length K+1) for the coordinator's equality accept-loop:
          greedy  -> r[j] = argmax(logits[j])      (identical to the legacy tail)
          sample  -> r[j] = d_j for accepted slots; r[m] = residual correction at first reject
                     (always != d_m); r[K] = bonus from p_K if all K accept."""
        K = len(draft)
        if self.greedy:
            return logits.argmax(-1).tolist()
        r = [0] * (K + 1)
        for j in range(K):
            p = self.dist(logits[j])
            dj = draft[j]
            if self.uniform() < float(p[dj]):
                r[j] = dj                              # accept the drafted token
            else:                                      # reject -> sample from the residual (d_j removed)
                p = p.clone(); p[dj] = 0.0
                s = float(p.sum())
                r[j] = self.sample(p / s) if s > 0 else int(self.dist(logits[j]).argmax().item())
                return r                               # first divergence: r[j] != d_j stops the accept-loop
        r[K] = self.sample(self.dist(logits[K]))       # all accepted -> one free target sample
        return r
