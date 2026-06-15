"""scheduler / control plane — light and replaceable.

fits the target model to the currently-joined (heterogeneous) gpus, orders them
into a pipeline preferring low-latency edges, tracks health, and reassigns blocks
when a node drops. holds no weights and no user data, so decentralizing it later
(rotating/elected) is a follow-up, not a rewrite. hosted by the c0mpute
orchestrator at first.
"""

from dataclasses import dataclass
from .node import LayerRange


@dataclass
class JoinedNode:
    node_id: str
    vram_gb: float
    rtt_ms: dict  # node_id -> measured rtt to other nodes


class Scheduler:
    def __init__(self, model: str, total_layers: int):
        self.model = model
        self.total_layers = total_layers
        self.nodes: dict[str, JoinedNode] = {}

    def register(self, node: JoinedNode) -> None:
        raise NotImplementedError  # phase 3

    def allocate(self) -> dict[str, LayerRange]:
        """assign each node a contiguous block that fits its vram, covering the stack.

        privacy note (docs/ARCHITECTURE.md#privacy): pin the embedding + final blocks
        to trusted/staked nodes, leave only deep middle blocks to untrusted volunteers.
        """
        raise NotImplementedError  # phase 0 (static) / phase 3 (dynamic+heterogeneous)

    def topology(self) -> list[str]:
        """order nodes into the pipeline, preferring low-rtt edges."""
        raise NotImplementedError  # phase 1

    def on_drop(self, node_id: str) -> None:
        """node died mid-generation: reassign its block, rebuild pipeline, retry in-flight."""
        raise NotImplementedError  # phase 3/4
