"""Wire NVFP4 execution via vLLM's real FusedMoE layer (modelopt nvfp4 method) — the piece
that unlocks ~16-node GLM-5.2. Constructs a standalone FusedMoE for GLM-5.2's MoE dims with
the NVFP4 quant config, loads layer-6's real NVFP4 experts through its weight_loader, runs
process_weights_after_loading (convert/swizzle/kernel setup), and forwards. Validate sane MoE
output; then integrate into the PP stage. run under /root/vmoe: python glm_nvfp4_moe.py
"""
import os, json, torch
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29577")
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("LOCAL_RANK", "0")
from safetensors import safe_open
from transformers import GlmMoeDsaConfig
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

DIR, dev, LAYER = "/root/glm52nvfp4", "cuda", 6
torch.cuda.set_device(0)
init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://", backend="nccl")
vcfg = VllmConfig(); _ctx = set_current_vllm_config(vcfg); _ctx.__enter__()   # vLLM layers need a current config
initialize_model_parallel(tensor_model_parallel_size=1)
init_workspace_manager(torch.device("cuda"))   # MoE kernel workspace allocator

cfg = GlmMoeDsaConfig.from_pretrained(DIR)
H, E, I = cfg.hidden_size, cfg.n_routed_experts, cfg.moe_intermediate_size
qcfg = ModelOptNvFp4Config.from_config(json.load(open(f"{DIR}/config.json"))["quantization_config"])
print(f"building FusedMoE: E={E} topk={cfg.num_experts_per_tok} H={H} I={I} nvfp4", flush=True)

# GLM-5.2 router = noaux_tc (sigmoid + group topk + correction bias + scaling)
moe = FusedMoE(
    num_experts=E, top_k=cfg.num_experts_per_tok, hidden_size=H, intermediate_size=I,
    params_dtype=torch.bfloat16, renormalize=cfg.norm_topk_prob,
    use_grouped_topk=True, num_expert_group=cfg.n_group, topk_group=cfg.topk_group,
    scoring_func="sigmoid", routed_scaling_factor=cfg.routed_scaling_factor,
    quant_config=qcfg, prefix="mlp.experts",
).to(dev)

# load the real NVFP4 experts via the layer's weight_loader
idx = json.load(open(f"{DIR}/model.safetensors.index.json"))["weight_map"]
_HD = {}
def raw(n):
    s = idx[n]
    if s not in _HD: _HD[s] = safe_open(f"{DIR}/{s}", "pt", device="cpu")
    return _HD[s].get_tensor(n)
P = f"model.layers.{LAYER}.mlp.experts."
params = dict(moe.named_parameters())
print("moe params:", [k for k in params if "weight" in k or "scale" in k], flush=True)

loaded = 0
for e in range(E):
    for proj, shard in [("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")]:
        grp = "w2" if shard == "w2" else "w13"
        for suf in ["weight", "weight_scale", "weight_scale_2", "input_scale"]:
            name = f"{P}{e}.{proj}.{suf}"
            if name not in idx:
                continue
            pname = f"{grp}_{suf}"
            if pname not in params:
                continue
            moe.weight_loader(params[pname], raw(name).to(dev), name, shard, e)
            loaded += 1
print(f"loaded {loaded} expert tensors into {sorted({('w2_'+s if d=='w2' else 'w13_'+s) for d in ['w1','w2'] for s in ['weight','weight_scale','weight_scale_2','input_scale'] if ('w2_'+s if d=='w2' else 'w13_'+s) in params})}", flush=True)

moe.quant_method.process_weights_after_loading(moe)
print("process_weights_after_loading OK -- nvfp4 kernel set up", flush=True)

torch.manual_seed(0)
T = 6
x = torch.randn(T, H, dtype=torch.bfloat16, device=dev) * 0.1
router_logits = torch.randn(T, E, dtype=torch.bfloat16, device=dev)
with torch.no_grad(), set_forward_context(None, vcfg):
    out = moe(x, router_logits)
print(f"\nNVFP4 FusedMoE forward: out {tuple(out.shape)} finite={torch.isfinite(out).all().item()} "
      f"mean|x| {out.abs().mean().item():.3f}", flush=True)
print("VERDICT:", "NVFP4 MoE EXECUTES via vLLM FusedMoE -- the 16-node kernel path works; integrate into PP stage."
      if torch.isfinite(out).all() else "ran but output not finite -- inspect.", flush=True)
