"""batched fast verify: the batch=1 fixed-shape CUDA-graph stage forward (fastverify.py),
lifted to B independent streams in ONE graph replay -- the engine primitive behind concurrent
request batching (DEPLOY_READINESS Â§2). throughput economics need >1 stream in flight; the
crux is that fastverify's StaticKV + decode graph are batch=1, fixed-shape.

what changes vs FastVerify (everything else -- MoE, sliding mask, bucketing -- is identical math):
  * StaticKV is [B, kv_heads, MAXLEN, head_dim]. each stream owns batch-row b; nothing crosses streams.
  * streams sit at DIFFERENT committed lengths (spec-decode commits a variable n+1 per round per stream),
    so the per-round write position is PER-STREAM: cp is [B, kp1] and the cache writes with a batched
    scatter_ (index_copy_ shares one index across the batch -> wrong for divergent starts). scatter at
    fixed shape is graph-capturable: we copy the round's positions into a persistent cp buffer, replay.
  * the additive causal/sliding mask is PER-STREAM: [B, 1, kp1, alen], query row qi of stream b holds
    abs pos start_b+qi. keys past start_b+qi (incl. the zero-init buffer tail) go to -inf -> 0 weight,
    so a shared bucket alen = bucket(max_b start_b + kp1) is bit-identical for every stream regardless
    of how far it has diverged (extra masked keys add exact 0.0 to the softmax sum).

bit-exactness: for B=1 this reduces to FastVerify (scatter of a contiguous range == index_copy_;
single-row mask == the [1,1,kp1,alen] mask). for B>1, stream b's output is bit-identical to running
that stream ALONE through FastVerify -- proven in research/batchverify_test.py on a real gpt-oss-120B
block (greedy argmax identical + hidden states equal). window-gather (FV_WINDOW) is intentionally NOT
ported: the bucketed full-buffer read is already bit-exact and is the engine default; sliding layers
still get their O(window) cost via the windowed-causal MASK, exactly like FastVerify's FV_WINDOW=0 path.
"""
import torch
from pipeline import _causal_mask
from fastverify import ContextOverflow, FastVerify


class BatchedStaticKV:
    """[B, kv_heads, MAXLEN, head_dim] K/V per layer. update() scatters this round's keys/values to
    each stream's own positions self.cp ([B, kp1]); the read returns the [:, :, :alen] window (the
    bucket covering the furthest-along stream) -- keys a given stream hasn't written are zero and the
    per-stream mask sends them to 0, so the batched read is bit-identical to each stream's solo read."""
    def __init__(self, n_layers, kv_heads, head_dim, maxlen, B, dev):
        z = lambda: [torch.zeros(B, kv_heads, maxlen, head_dim, dtype=torch.bfloat16, device=dev)
                     for _ in range(n_layers)]
        self.k, self.v = z(), z()
        self.cp = None                                           # [B, kp1] per-stream write positions
        self.alen = maxlen
        self.B = B; self.kv_heads = kv_heads; self.head_dim = head_dim

    def update(self, key, value, layer_idx, *a, **kw):
        # key/value: [B, kv_heads, kp1, head_dim]. scatter to K[b, :, cp[b,j], :] = key[b, :, j, :].
        kp1 = key.shape[2]
        idx = self.cp.view(self.B, 1, kp1, 1).expand(self.B, self.kv_heads, kp1, self.head_dim)
        self.k[layer_idx].scatter_(2, idx, key)
        self.v[layer_idx].scatter_(2, idx, value)
        return self.k[layer_idx][:, :, :self.alen, :], self.v[layer_idx][:, :, :self.alen, :]


def _batched_causal_mask(kp1, alen, starts, window, dtype, device):
    """additive [B, 1, kp1, alen] mask. query row qi of stream b is at abs pos starts[b]+qi; it attends
    key col j (abs pos j) iff j <= starts[b]+qi and -- when window>0 -- also (starts[b]+qi - j) < window."""
    rows = torch.arange(kp1, device=device)[None, :] + starts[:, None]    # [B, kp1] abs query positions
    cols = torch.arange(alen, device=device)                             # [alen] abs key positions
    allow = cols[None, None, :] <= rows[:, :, None]                      # [B, kp1, alen] causal
    if window:
        allow = allow & ((rows[:, :, None] - cols[None, None, :]) < window)
    minv = torch.finfo(dtype).min
    return torch.where(allow, torch.zeros((), dtype=dtype, device=device),
                       torch.full((), minv, dtype=dtype, device=device))[:, None]


class BatchedFastVerify:
    """B-stream version of FastVerify.decode: one fixed-shape CUDA graph verifies a K+1 chunk for every
    stream at once. Streams may sit at arbitrary, divergent committed lengths (starts is [B]). Only the
    graphed decode path is provided (the steady-state, throughput-critical op); prefill is done per
    stream into batch-row b via load_prefix() (copies a solo FastVerify's committed KV), keeping the
    proven flex prefill untouched -- batching is a DECODE-time concern."""
    DECODE_BUCKETS = FastVerify.DECODE_BUCKETS

    def __init__(self, parts, B, maxlen=2048, dev="cuda", lossless=False):
        self.parts = parts; self.B = B; self.maxlen = maxlen; self.dev = dev
        # lossless: the mxfp4 MoE kernel is NOT token-count-invariant, so a stacked batch dim perturbs each
        # stream's MoE output (the cross-K FP class). attention IS batch-invariant (eager matmul+softmax+sink),
        # so we batch attention but run the MoE PER STREAM (kp1 tokens each, exactly as solo) -> bit-exact per
        # stream, at the cost of the MoE weight-reuse win. wraps the SHARED layer.mlp (identity at B<=1, so the
        # reference FastVerify is unaffected). Off by default (the fast, FP-lossy path).
        self.lossless = lossless
        if lossless:
            for layer in parts["layers"]:
                self._wrap_mlp(layer)
        self.layers = parts["layers"]; self.n_layers = len(self.layers)
        self.sliding = parts.get("sliding"); self.win = parts.get("window", 0)
        self.rotary = parts["rotary"]
        cfg = parts["_model"].config
        self.hidden = cfg.hidden_size
        kvh = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
        hd = getattr(cfg, "head_dim", None) or (self.hidden // cfg.num_attention_heads)
        self.kvh = kvh; self.hd = hd
        self.cache = BatchedStaticKV(self.n_layers, kvh, hd, maxlen, B, dev)
        self.cfg = cfg
        self.cfg._attn_implementation = "eager"                  # decode is graphed eager (q=K+1 cheap over 100k kv)
        self.kp1 = None; self.graph = None; self.out = None
        self.alen_built = None

    def _wrap_mlp(self, layer):
        """wrap a decoder layer's MoE mlp so a [B,S,H] batch is processed PER STREAM ([1,S,H] each) — the
        token count the kernel sees per stream then matches solo decode, so the MoE is bit-exact. Identity
        at B<=1, so the shared reference FastVerify is unaffected. idempotent (won't double-wrap)."""
        mlp = layer.mlp
        if getattr(mlp, "_bv_wrapped", False):
            return
        orig = mlp.forward
        def wrapped(x, *a, **k):
            if x.shape[0] <= 1:
                return orig(x, *a, **k)
            outs = [orig(x[b:b + 1], *a, **k) for b in range(x.shape[0])]
            if isinstance(outs[0], tuple):
                return tuple(torch.cat([o[i] for o in outs], 0) for i in range(len(outs[0])))
            return torch.cat(outs, 0)
        mlp.forward = wrapped
        mlp._bv_wrapped = True

    def _bucket(self, need):
        for b in self.DECODE_BUCKETS:
            if b >= need:
                return min(b, self.maxlen)
        return self.maxlen

    def reset(self):
        for t in self.cache.k: t.zero_()
        for t in self.cache.v: t.zero_()

    def load_prefix(self, b, src_cache):
        """copy stream b's committed KV (a solo FastVerify's StaticKV, batch-1) into batch-row b. used
        to seat each stream's prefill into the shared batched cache before batched decoding starts."""
        for l in range(self.n_layers):
            self.cache.k[l][b].copy_(src_cache.k[l][0])
            self.cache.v[l][b].copy_(src_cache.v[l][0])

    def _layers(self, x, pos, pe, mf, mw):
        for i, layer in enumerate(self.layers):
            m = mw if (self.sliding and self.sliding[i]) else mf
            o = layer(x, attention_mask=m, position_ids=pos, past_key_values=self.cache,
                      use_cache=True, position_embeddings=pe)
            x = o[0] if isinstance(o, tuple) else o
        return x

    def _build(self, kp1, alen):
        self.kp1 = kp1; self.alen_built = alen
        B = self.B
        self.h_buf = torch.zeros(B, kp1, self.hidden, dtype=torch.bfloat16, device=self.dev)
        self.pos_buf = torch.zeros(B, kp1, dtype=torch.long, device=self.dev)
        self.cp_buf = torch.zeros(B, kp1, dtype=torch.long, device=self.dev)
        self.mf_buf = torch.zeros(B, 1, kp1, alen, dtype=torch.bfloat16, device=self.dev)
        self.mw_buf = torch.zeros(B, 1, kp1, alen, dtype=torch.bfloat16, device=self.dev)

    def _set(self, h, starts):
        """h: [B, kp1, hidden]; starts: long [B] per-stream committed length (write origin)."""
        self.h_buf.copy_(h)
        ar = torch.arange(self.kp1, device=self.dev)[None, :] + starts[:, None]    # [B, kp1] abs positions
        self.pos_buf.copy_(ar); self.cp_buf.copy_(ar)
        self.mf_buf.copy_(_batched_causal_mask(self.kp1, self.alen_built, starts, 0, torch.bfloat16, self.dev))
        if self.win:
            self.mw_buf.copy_(_batched_causal_mask(self.kp1, self.alen_built, starts, self.win, torch.bfloat16, self.dev))
        else:
            self.mw_buf.copy_(self.mf_buf)

    def _graph_body(self):
        self.cache.cp = self.cp_buf
        self.cache.alen = self.alen_built
        return self._layers(self.h_buf, self.pos_buf, self.rotary(self.h_buf, self.pos_buf),
                            self.mf_buf, self.mw_buf)

    def decode(self, h, starts, use_graph=True):
        """h: [B, kp1, hidden]; starts: long tensor [B]. returns [B, kp1, hidden]. One captured graph
        replays all B streams; rebuilds only when kp1 or the (shared) bucket changes. use_graph=False
        runs the same math eagerly (debug / kernels that aren't CUDA-graph-safe at B>1)."""
        kp1 = h.shape[1]
        need = int(starts.max().item()) + kp1
        if need > self.maxlen:
            raise ContextOverflow(f"batched decode needs {need} positions > max_ctx {self.maxlen}")
        alen = self._bucket(need)                                # shared bucket = furthest stream's span
        if self.kp1 != kp1 or self.alen_built != alen:
            self._build(kp1, alen); self.graph = None
        self._set(h, starts)
        if not use_graph:
            return self._graph_body()
        if self.graph is None:
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): self._graph_body()
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self._graph_body()
            self._set(h, starts)                                 # warmup dirtied the cache; restore round inputs
        self.graph.replay()
        return self.out
