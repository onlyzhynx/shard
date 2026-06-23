"""Is the gpt-oss block batch-invariant? Feed identical token rows at B=1 vs B=2 through (a) the MoE
MLP alone and (b) the whole decoder layer (no cache), compare row 0. If the MLP diverges, the mxfp4
triton MoE kernel is not per-token batch-invariant -> batched verify must pack tokens, not stack a
batch dim."""
import argparse, torch
from pipeline import load_stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/gpt-oss-120b")
    ap.add_argument("--stage", type=int, default=8)
    ap.add_argument("--nstages", type=int, default=9)
    a = ap.parse_args()
    dev = "cuda"
    parts = load_stage(a.model, a.stage, a.nstages, device=dev)
    H = parts["_model"].config.hidden_size
    layer = parts["layers"][0]
    torch.manual_seed(0)
    x1 = torch.randn(1, 5, H, dtype=torch.bfloat16, device=dev) * 0.1

    with torch.no_grad():
        # --- MoE MLP alone ---
        mlp = layer.mlp
        o1 = mlp(x1)
        o1 = o1[0] if isinstance(o1, tuple) else o1
        for B in (2, 4, 8):
            xb = x1.expand(B, 5, H).contiguous()
            ob = mlp(xb)
            ob = ob[0] if isinstance(ob, tuple) else ob
            d = (ob[0].float() - o1[0].float()).abs().max().item()
            intra = (ob[0].float() - ob[B - 1].float()).abs().max().item()
            print(f"[MLP] B={B}: max|row0(B)-row0(1)|={d:.4g}  intra|row0-row{B-1}|={intra:.4g}", flush=True)

        # --- whole layer, fresh (no cache), causal mask ---
        from pipeline import _causal_mask
        def run_layer(x):
            B, S, _ = x.shape
            pos = torch.arange(S, device=dev).unsqueeze(0).expand(B, S)
            pe = parts["rotary"](x, pos)
            m = _causal_mask(S, S, 0, 0, x.dtype, dev).expand(B, 1, S, S)
            o = layer(x, attention_mask=m, position_ids=pos, use_cache=False, position_embeddings=pe)
            return o[0] if isinstance(o, tuple) else o
        l1 = run_layer(x1)
        for B in (2, 4):
            lb = run_layer(x1.expand(B, 5, H).contiguous())
            d = (lb[0].float() - l1[0].float()).abs().max().item()
            print(f"[LAYER] B={B}: max|row0(B)-row0(1)|={d:.4g}", flush=True)


if __name__ == "__main__":
    main()
