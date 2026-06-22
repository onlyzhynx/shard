"""Single-GPU long-context benchmark for Shard's FastVerify path.

The module-level helpers intentionally use only the Python standard library so
tests can import them on machines without CUDA, torch, or transformers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence


DEFAULT_CONTEXTS = [2048, 4096, 8192, 12000, 16000, 20000, 24000, 32768]
MODES = ("baseline", "fv_window")
REQUIRED_FIELDS = {
    "timestamp",
    "git_commit",
    "model_id",
    "mode",
    "ctx_len",
    "max_ctx",
    "K",
    "fast_enabled",
    "fv_window_enabled",
    "tok_s",
    "verify_ms",
    "draft_ms",
    "tokens_per_traversal",
    "mean_accept",
    "n_tokens",
    "rounds",
    "gpu_memory_allocated_mb",
    "gpu_memory_reserved_mb",
    "output_ids",
    "output_ids_sha256",
    "output_ids_match_baseline",
    "first_differing_token_index",
    "baseline_output_ids_sha256",
    "test_output_ids_sha256",
}


def output_ids_sha256(output_ids: Sequence[int]) -> str:
    """Return a representation-independent hash of an integer token sequence."""
    payload = json.dumps(list(output_ids), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def compare_output_ids(
    baseline_ids: Sequence[int], test_ids: Sequence[int]
) -> tuple[bool, int | None]:
    """Return exact equality and the first differing index, including length differences."""
    for index, (baseline_id, test_id) in enumerate(zip(baseline_ids, test_ids)):
        if baseline_id != test_id:
            return False, index
    if len(baseline_ids) != len(test_ids):
        return False, min(len(baseline_ids), len(test_ids))
    return True, None


def validate_result_row(row: dict[str, Any]) -> None:
    missing = REQUIRED_FIELDS.difference(row)
    if missing:
        raise ValueError(f"result row is missing required fields: {sorted(missing)}")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
            validate_result_row(row)
            rows.append(row)
    return rows


def baseline_outputs_from_rows(rows: Iterable[dict[str, Any]]) -> dict[int, list[int]]:
    outputs: dict[int, list[int]] = {}
    for row in rows:
        if row.get("mode") != "baseline" or "output_ids" not in row:
            continue
        if row.get("record_type", "summary") == "summary":
            outputs[int(row["ctx_len"])] = [int(token_id) for token_id in row["output_ids"]]
    if outputs:
        return outputs
    for row in rows:
        if row.get("mode") == "baseline" and "output_ids" in row and not row.get("warmup", False):
            outputs[int(row["ctx_len"])] = [int(token_id) for token_id in row["output_ids"]]
    return outputs


def synthetic_prompt_ids(tokenizer: Any, ctx_len: int) -> list[int]:
    """Build exactly ``ctx_len`` deterministic IDs from varied, neutral prose."""
    if ctx_len <= 0:
        raise ValueError("ctx_len must be positive")
    text = (
        "A quiet library keeps maps, notebooks, and reference cards in orderly rows. "
        "Researchers compare measurements, record assumptions, and repeat each trial carefully. "
        "Morning light crosses the tables while a clock marks steady intervals. "
        "The notes describe ordinary weather, public gardens, simple machines, and clear water. "
        "Each paragraph changes its wording so the context contains a varied token pattern. "
    )
    seed_ids = tokenizer.encode(text, add_special_tokens=False)
    if not seed_ids:
        raise ValueError("tokenizer produced no IDs for the synthetic prompt")
    repeats = (ctx_len + len(seed_ids) - 1) // len(seed_ids)
    return (seed_ids * repeats)[:ctx_len]


def deterministic_drafts(source_ids: Sequence[int], current_id: int, K: int, round_index: int) -> list[int]:
    """Produce cheap, varied candidates; target correction keeps generation greedy-exact."""
    if K <= 0:
        return []
    span = min(len(source_ids), 257)
    if span == 0:
        return [current_id] * K
    offset = (round_index * (K + 3) + current_id) % span
    return [int(source_ids[(offset + index * 17) % span]) for index in range(K)]


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _comparison_fields(output_ids: Sequence[int], baseline_ids: Sequence[int] | None) -> dict[str, Any]:
    output_hash = output_ids_sha256(output_ids)
    if baseline_ids is None:
        return {
            "output_ids_sha256": output_hash,
            "output_ids_match_baseline": None,
            "first_differing_token_index": None,
            "baseline_output_ids_sha256": None,
            "test_output_ids_sha256": output_hash,
        }
    matches, first_difference = compare_output_ids(baseline_ids, output_ids)
    return {
        "output_ids_sha256": output_hash,
        "output_ids_match_baseline": matches,
        "first_differing_token_index": first_difference,
        "baseline_output_ids_sha256": output_ids_sha256(baseline_ids),
        "test_output_ids_sha256": output_hash,
    }


def _base_row(args: argparse.Namespace, mode: str, ctx_len: int, commit: str | None) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "model_id": args.model,
        "mode": mode,
        "ctx_len": ctx_len,
        "max_ctx": args.max_ctx,
        "K": args.K,
        "device": args.device,
        "max_new": args.max_new,
        "prefill_chunk": args.prefill_chunk,
        "fast_enabled": True,
        "fv_window_enabled": mode == "fv_window",
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    validate_result_row(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def summarize_rows(
    rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    mode: str,
    ctx_len: int,
    commit: str | None,
    baseline_ids: Sequence[int] | None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize zero measured runs")
    output_ids = rows[-1]["output_ids"]
    summary = _base_row(args, mode, ctx_len, commit)
    summary.update(
        {
            "record_type": "summary",
            "warmup": False,
            "repeat_index": None,
            "tok_s": _mean(rows, "tok_s"),
            "verify_ms": _mean(rows, "verify_ms"),
            "draft_ms": _mean(rows, "draft_ms"),
            "tokens_per_traversal": _mean(rows, "tokens_per_traversal"),
            "mean_accept": _mean(rows, "mean_accept"),
            "n_tokens": round(_mean(rows, "n_tokens")),
            "rounds": round(_mean(rows, "rounds")),
            "gpu_memory_allocated_mb": max(row["gpu_memory_allocated_mb"] for row in rows),
            "gpu_memory_reserved_mb": max(row["gpu_memory_reserved_mb"] for row in rows),
            "output_ids": output_ids,
            "output_ids_consistent": all(row["output_ids"] == output_ids for row in rows),
            "repeat": len(rows),
        }
    )
    summary.update(_comparison_fields(output_ids, baseline_ids))
    return summary


def run_generation(
    *,
    torch: Any,
    parts: dict[str, Any],
    fast_verify: Any,
    prompt_ids: Sequence[int],
    max_new: int,
    K: int,
    device: str,
    prefill_chunk: int,
) -> dict[str, Any]:
    """Run local fixed-K greedy verification; prefill time is intentionally excluded."""
    fast_verify.reset()
    with torch.inference_mode():
        last_hidden = None
        for start in range(0, len(prompt_ids), prefill_chunk):
            chunk = prompt_ids[start : start + prefill_chunk]
            token_ids = torch.tensor([chunk], dtype=torch.long, device=device)
            hidden = parts["embed"](token_ids)
            last_hidden = fast_verify.prefill(hidden, start)
        if last_hidden is None:
            raise ValueError("prompt must contain at least one token")
        logits = parts["lm_head"](parts["norm"](last_hidden[:, -1:, :]))
        current_id = int(logits[0, -1].argmax().item())

        output_ids = [current_id]
        position = len(prompt_ids)
        rounds = 0
        accepted_total = 0
        draft_seconds = 0.0
        verify_seconds = 0.0
        decode_started = time.perf_counter()

        while len(output_ids) < max_new:
            draft_started = time.perf_counter()
            drafts = deterministic_drafts(prompt_ids, current_id, K, rounds)
            draft_seconds += time.perf_counter() - draft_started

            verify_input = [current_id, *drafts]
            hidden = parts["embed"](torch.tensor([verify_input], dtype=torch.long, device=device))
            torch.cuda.synchronize(device)
            verify_started = time.perf_counter()
            verified_hidden = fast_verify.decode(hidden, position)
            verified_logits = parts["lm_head"](parts["norm"](verified_hidden))
            predictions = verified_logits.argmax(-1)[0].tolist()
            torch.cuda.synchronize(device)
            verify_seconds += time.perf_counter() - verify_started

            accepted = 0
            for draft_id, prediction_id in zip(drafts, predictions):
                if draft_id != prediction_id:
                    break
                accepted += 1
            committed = drafts[:accepted] + [int(predictions[accepted])]
            output_ids.extend(committed)
            current_id = committed[-1]
            position += len(committed)
            accepted_total += accepted
            rounds += 1

        decode_seconds = time.perf_counter() - decode_started

    output_ids = output_ids[:max_new]
    return {
        "tok_s": len(output_ids) / max(decode_seconds, 1e-9),
        "verify_ms": verify_seconds / max(rounds, 1) * 1000.0,
        "draft_ms": draft_seconds / max(rounds, 1) * 1000.0,
        "tokens_per_traversal": (accepted_total + rounds) / max(rounds, 1),
        "mean_accept": accepted_total / max(rounds, 1),
        "n_tokens": len(output_ids),
        "rounds": rounds,
        "gpu_memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
        "gpu_memory_reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        "output_ids": output_ids,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ctx", nargs="+", type=int, default=DEFAULT_CONTEXTS)
    parser.add_argument("--mode", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--max-ctx", type=int, default=None)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prefill-chunk", type=int, default=2048)
    parser.add_argument("--out", default="results/context_sweep.jsonl")
    parser.add_argument("--baseline-file", default=None)
    parser.add_argument("--log-warmup", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    if any(ctx_len <= 0 for ctx_len in args.ctx):
        parser.error("--ctx values must be positive")
    if args.K < 1:
        parser.error("--K must be at least 1")
    if args.max_new < 1 or args.repeat < 1 or args.warmup < 0 or args.prefill_chunk < 1:
        parser.error("--max-new, --repeat, and --prefill-chunk must be positive; --warmup cannot be negative")
    required_max_ctx = max(args.ctx) + args.max_new + args.K
    if args.max_ctx is None:
        args.max_ctx = required_max_ctx
    elif args.max_ctx < required_max_ctx:
        parser.error(
            f"--max-ctx {args.max_ctx} is too small; need at least {required_max_ctx} "
            "for the largest prompt plus generation and verify scratch"
        )
    args.mode = [mode for mode in MODES if mode in args.mode]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.device.startswith("cuda"):
        raise SystemExit("context_sweep requires a CUDA device")

    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit(f"benchmark dependencies are unavailable: {error}") from error
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; utility tests remain CPU-only")

    phase0_path = Path(__file__).resolve().parents[1] / "phase0"
    sys.path.insert(0, str(phase0_path))
    from fastverify import FastVerify
    from pipeline import load_stage

    output_path = Path(args.out)
    commit = git_commit()
    baseline_outputs: dict[int, list[int]] = {}
    if args.baseline_file:
        baseline_outputs.update(baseline_outputs_from_rows(read_jsonl(args.baseline_file)))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = {ctx_len: synthetic_prompt_ids(tokenizer, ctx_len) for ctx_len in args.ctx}
    parts = load_stage(args.model, 0, 1, device=args.device, attn="eager")

    print(
        f"context sweep: model={args.model} device={args.device} max_ctx={args.max_ctx} "
        f"K={args.K} contexts={args.ctx}",
        flush=True,
    )
    for mode in args.mode:
        if mode == "baseline":
            os.environ.pop("FV_WINDOW", None)
        else:
            os.environ["FV_WINDOW"] = "1"
        fast_verify = FastVerify(parts, maxlen=args.max_ctx, dev=args.device)
        try:
            for ctx_len in args.ctx:
                measured_rows = []
                baseline_ids = baseline_outputs.get(ctx_len)
                total_runs = args.warmup + args.repeat
                for run_index in range(total_runs):
                    warmup = run_index < args.warmup
                    metrics = run_generation(
                        torch=torch,
                        parts=parts,
                        fast_verify=fast_verify,
                        prompt_ids=prompts[ctx_len],
                        max_new=args.max_new,
                        K=args.K,
                        device=args.device,
                        prefill_chunk=args.prefill_chunk,
                    )
                    row = _base_row(args, mode, ctx_len, commit)
                    row.update(metrics)
                    row.update(
                        {
                            "record_type": "run",
                            "warmup": warmup,
                            "repeat_index": run_index - args.warmup if not warmup else run_index,
                        }
                    )
                    comparison_baseline = metrics["output_ids"] if mode == "baseline" else baseline_ids
                    row.update(_comparison_fields(metrics["output_ids"], comparison_baseline))
                    if not warmup:
                        measured_rows.append(row)
                    if not warmup or args.log_warmup:
                        _append_jsonl(output_path, row)
                    print(
                        f"[{mode} ctx={ctx_len} {'warmup' if warmup else f'run={run_index - args.warmup + 1}'}] "
                        f"{metrics['tok_s']:.2f} tok/s verify={metrics['verify_ms']:.2f}ms "
                        f"accept={metrics['mean_accept']:.2f}",
                        flush=True,
                    )

                if mode == "baseline":
                    baseline_outputs[ctx_len] = list(measured_rows[-1]["output_ids"])
                    baseline_ids = baseline_outputs[ctx_len]
                summary = summarize_rows(measured_rows, args, mode, ctx_len, commit, baseline_ids)
                _append_jsonl(output_path, summary)
                print(
                    f"[{mode} ctx={ctx_len} summary] {summary['tok_s']:.2f} tok/s "
                    f"verify={summary['verify_ms']:.2f}ms match={summary['output_ids_match_baseline']}",
                    flush=True,
                )
        finally:
            del fast_verify
            torch.cuda.empty_cache()

    os.environ.pop("FV_WINDOW", None)
    print(f"wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
