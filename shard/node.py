"""per-node runtime: serve one contiguous block of layers via SGLang.

we do not rebuild kernels/attention/paging. this wraps SGLang and drives it at
block granularity. a node = a block of layers + a transport endpoint + a heartbeat.
"""

from dataclasses import dataclass


@dataclass
class LayerRange:
    start: int  # inclusive
    end: int    # exclusive


class NodeRuntime:
    """wraps an SGLang engine loaded with only [layer_range] of `model`."""

    def __init__(self, model: str, layer_range: LayerRange, device: str = "cuda:0"):
        self.model = model
        self.layer_range = layer_range
        self.device = device

    def load_shard(self) -> None:
        """pull only this block's weights (hf or c0mpute mirror) and load to vram."""
        raise NotImplementedError  # phase 0

    def forward(self, hidden_states, kv_meta):
        """run this block's forward, return activations for the next stage.

        hidden_states: input activations from the previous stage (or embeddings if first).
        kv_meta: sequence ids / positions so this node manages its own kv-cache.
        returns: activations to hand to the next stage (or logits if last).
        """
        raise NotImplementedError  # phase 0

    def heartbeat(self) -> dict:
        """liveness + vram/util, reported to the scheduler."""
        raise NotImplementedError
