"""speculative decoding coordinator — what makes wan latency survivable.

entry node runs a small draft model locally (no wan cost), proposes K tokens,
the big split target verifies all K in ONE pipeline traversal. accepted tokens
are kept, the first rejection is resampled. net: ~several accepted tokens per
round-trip instead of one, which amortizes the wan latency.

based on arxiv 2602.16760. K is tuned to measured link latency + live acceptance.
"""


class DraftModel:
    """small (1-3B), uncensored, same tokenizer family as the target. runs local."""

    def propose(self, context, k: int):
        """return k candidate tokens + their draft probabilities."""
        raise NotImplementedError  # phase 2


class SpeculativeLoop:
    """drives propose -> verify-across-swarm -> accept, with adaptive K."""

    def __init__(self, draft: "DraftModel", pipeline, init_k: int = 4):
        self.draft = draft
        self.pipeline = pipeline  # the split target, exposed as one verify() call
        self.k = init_k

    def verify(self, context, draft_tokens):
        """one pipeline traversal: target distributions for all draft positions."""
        raise NotImplementedError  # phase 2

    def step(self, context):
        """one round: propose K locally, verify across the swarm, accept the prefix."""
        raise NotImplementedError  # phase 2

    def tune_k(self, measured_rtt_ms: float, acceptance_rate: float) -> int:
        """higher latency / higher acceptance -> draft deeper to amortize more."""
        raise NotImplementedError  # phase 2
