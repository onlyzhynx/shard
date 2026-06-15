"""inter-stage transport — the part we own, the wedge.

moves the activation tensor across the public internet between two adjacent
pipeline stages, reliably, over nat. quic + hole-punching + relay fallback +
a quantized activation codec + backpressure/reconnect. fully instrumented:
every edge logs its own health, no opaque "broken pipe".

start: aioquic in python. lift the hot path to rust (quinn) when bandwidth demands.
"""


class ActivationCodec:
    """serialize + (fp8/int8) quantize + optionally compress a hidden-state tensor.

    activations tolerate quantization far better than weights, and bandwidth is the
    cost center on a home uplink, so this is where we spend effort.
    """

    def encode(self, hidden_states) -> bytes:
        raise NotImplementedError  # phase 1

    def decode(self, payload: bytes):
        raise NotImplementedError  # phase 1


class Edge:
    """one supervised pipeline edge: a quic connection from this stage to the next.

    handles hole-punch via the rendezvous, relay fallback, send/recv of encoded
    activations, timeouts, backpressure, reconnect. surfaces health() so a stalled
    edge is detected fast instead of wedging the pipeline.
    """

    def __init__(self, peer_id: str, rendezvous: str, codec: "ActivationCodec"):
        self.peer_id = peer_id
        self.rendezvous = rendezvous
        self.codec = codec

    async def connect(self) -> None:
        """hole-punch to peer via rendezvous; fall back to relay on symmetric nat."""
        raise NotImplementedError  # phase 0 (direct) / phase 1 (nat)

    async def send(self, hidden_states) -> None:
        raise NotImplementedError  # phase 0

    async def recv(self):
        raise NotImplementedError  # phase 0

    def health(self) -> dict:
        """rtt, in-flight, last-ok timestamp. the thing the black-box binary never gave us."""
        raise NotImplementedError
