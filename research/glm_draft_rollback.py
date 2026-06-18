"""Isolate StaticCache rollback correctness for the draft (no swarm, eager). The pipe rewinds the
write position on divergence; StaticCache leaves the rejected drafts in the buffer. Question: does a
re-draft after rewind match a clean fresh draft? Test 3 strategies for masking the stale tail:
  none  - pass nothing (current bug: HF causal mask keys off max-written -> attends stale)
  mask  - attention_mask[0..pos]=1 (what I tried; broke it)
  zero  - physically zero the stale K/V slots in the buffer on rewind
  python glm_draft_rollback.py"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
DRAFT = "/root/glm4_9b_draft"; dev = "cuda"

tok = AutoTokenizer.from_pretrained(DRAFT, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
ids = tok("def quicksort(arr):", return_tensors="pt").input_ids.to(dev); L = ids.shape[1]
N = 16; MAXLEN = L + 64

def new_cache():
    c = StaticCache(config=m.config, max_cache_len=MAXLEN, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        cur = int(m(input_ids=ids, past_key_values=c, cache_position=torch.arange(L, device=dev), use_cache=True).logits[0, -1].argmax())
    return c, cur

def step(c, t, p, strategy):
    inp = torch.tensor([[t]], device=dev); cp = torch.tensor([p], device=dev)
    kw = {}
    if strategy == "mask":
        mk = torch.zeros((1, MAXLEN), dtype=torch.long, device=dev); mk[0, :p + 1] = 1; kw["attention_mask"] = mk
    if strategy == "maskpos":   # mask the stale tail AND pin position_ids (else HF derives them from the mask cumsum -> RoPE breaks)
        mk = torch.zeros((1, MAXLEN), dtype=torch.long, device=dev); mk[0, :p + 1] = 1
        kw["attention_mask"] = mk; kw["position_ids"] = cp.unsqueeze(0)
    with torch.no_grad():
        return int(m(input_ids=inp, past_key_values=c, cache_position=cp, use_cache=True, **kw).logits[0, -1].argmax())

# 1. clean fresh draft of N tokens
c, cur = new_cache(); fresh = []; t = cur
for i in range(N): t = step(c, t, L + i, "none"); fresh.append(t)
print("fresh:", fresh, flush=True)

# 2. rollback: draft 8, rewind to L+4 (drop tokens 4..7), re-draft -> should match fresh[4:]
for strat in ["none", "mask", "maskpos"]:
    c, cur = new_cache(); t = cur; got = []
    for i in range(8): t = step(c, t, L + i, strat); got.append(t)   # draft 0..7
    cut = 4
    if strat == "zero":                                              # physically wipe stale K/V at [L+cut ..]
        for layer in c.layers:
            if layer.keys is not None:
                layer.keys[:, :, L + cut:, :] = 0; layer.values[:, :, L + cut:, :] = 0
    t = fresh[cut - 1]; redraft = []                                 # re-feed token at L+cut-1 (its own slot), predict L+cut onward
    for p in range(L + cut - 1, L + N - 1): t = step(c, t, p, strat); redraft.append(t)
    match = sum(1 for a, b in zip(redraft, fresh[cut:]) if a == b)
    print(f"{strat:5s}: redraft[:6]={redraft[:6]} | matches fresh[{cut}:] = {match}/{len(fresh)-cut}", flush=True)
