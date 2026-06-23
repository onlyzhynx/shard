# Shard fleet — 2026-06-23

## Session 3 (batched verify + async inter-stage send) — LIVE
Goal: #1 concurrent/continuous request batching (batched fast-verify CUDA graph) + #2 async
inter-stage send to cash in pipelined-prefill TTFT at 100k. Rented 6 distinct-host scattered US
4090s (cuda-13.2.1-auto, ALL with `--env '-p 29600:29600'`), genuinely scattered states:
| role | id | geo | ip | ssh | :29600 |
|------|-----|-----|-----|-----|--------|
| il (head) | 42248154 | Illinois | 104.12.231.85 | 40206 | 40242 |
| nv | 42248163 | Nevada | 173.239.95.142 | 41359 | 41370 |
| mi | 42248167 | Michigan | 216.234.102.170 | 14391 | 14482 |
| nj | 42248170 | New Jersey | 71.104.167.38 | 50596 | 50662 |
| ca | 42248173 | California | 192.234.50.153 | 3306 | 3323 |
| wa | 42248182 | Washington | 50.175.95.210 | 50232 | 50125 |
~$2.5/hr total. 6 distinct machine_ids (IL/NV/MI/NJ/CA/WA). Bootstrap: fleet.py push
setup_box.sh,get_model.py,stage_bootstrap.sh → fire stage_bootstrap detached → poll stage_ready.txt.
N=4 ring for the #2 baseline (vs old 193s@110k); N=6 to chase <60s. One box doubles as the #1
single-box batched-verify dev box. TEARDOWN: `vastai destroy instance <id>` per box when done.

RESULTS (committed): #1 batched verify proven on CA box (B=1 bit-exact, 1.6×@B4/2.1×@B8 throughput;
MoE token-count non-invariance isolated). #2 async-send A/B on the N=4 ring (IL·NV·CA·NJ):
30k 153.3→60.8s (2.52×), 110k 245.9→210.0s (1.17×, compute-bound). N=6 (MI+WA) and #3 hot-standby
NOT run (budget; documented as next levers). Fleet TORN DOWN after the commit (no idle spend).

## Session 2 (deploy-readiness: sampling / TTFT / fault tolerance) — TORN DOWN
Rented 5 distinct-host scattered US 4090s (cuda-13.2.1-auto, `-p 29600:29600`): WA·MN·NC·NJ ring (even
N=4, 9 layers/box) + OH hot-spare. Used for: lossless speculative sampling (DONE), pipelined-prefill TTFT
A/B (partial), mid-request fault tolerance (demonstrated). All 5 destroyed when done — see git commits
`f3fba2d` (engine) + `39fdd5b` (receipts/docs). New tooling this session:
- `specpipe --temp/--top-p/--top-k/--seed` → lossless sampling; `--sample-test N` → on-swarm losslessness proof.
- `specpipe --prefill-depth D` → pipelined prefill (overlap chunks across stages).
- `phase0/heal.py --ring … --spare … --kill-stage k` → mid-request fault-tolerance demo (kill→heal→resume).
- `research/specsample_proof.py` → local (no-GPU) losslessness proof of the acceptance math.
Bring-up was `launch_ngram.py` with even split (omit `--layers`); coordinator on the head box.
Even N=4 (9/9/9/9 on four 24GB 4090s) fits ~110k KV with room — faster than the old heterogeneous 18/9/9
(110k prefill 226.8s vs ~556s). NOTE: still create boxes WITH `-p 29600:29600` (vast-expose-29600 memory).

## Session 1 (long-context perf) — TORN DOWN
Goal: ≥20 tok/s on >100k context, swarm on ≤4 neighbouring western states, trustworthy output.

## Instances (vast, account=leyten, key ~/.ssh/vast_c0mpute, image cuda-13.2.1-auto)
All DISTINCT host_id (no co-location). Created WITH `--env '-p 29600:29600'` (inter-stage
transport port — REQUIRED, see vast-expose-29600-port memory). N=3: the cuda-13.2 western pool
had only 3 distinct usable hosts (both CA hosts broke — 224600 ssh-key, 392559 docker-pull).
VRAM-aware uneven split: 48GB box holds 18 layers, two 24GB boxes 9 each (load_stage --lo/--hi).
NO separate coord box: model-free n-gram coordinator is CPU-only, runs ON the head box.
| id | label | role | host_id | VRAM | layers | $/hr |
|----|-------|------|---------|------|--------|------|
| 42195546 | shard-stage-wa2 | stage0 (head) + coordinator | 96690 | 48GB | [0:18] | 1.120 |
| 42195544 | shard-stage-wa1 | stage1 | 22965 | 24GB | [18:27] | 0.362 |
| 42195547 | shard-stage-tx | stage2 (tail) | 558496 | 24GB | [27:36] | 0.336 |

~$1.82/hr total. Ring: wa2(head,18L) -> wa1(9L) -> tx(tail,9L) -> return to wa2. 2 states (WA,TX).
STAGES=42195546,42195544,42195547  layers=18,9,9  head=42195546
Launch (brings up ring + runs ngram long-ctx coordinator):
  cd phase0 && SHARD_PSK=$(cat ~/.shard_psk) python3 launch_ngram.py --stages 42195546,42195544,42195547 --layers 18,9,9 --max-ctx 131072 --prompt-file /root/prompt_long.txt --prefill-chunk 4096 --depth 4 --K 4 --ngram-n 3 --max-new 256
Re-run coordinator on warm ring: add --no-launch. Window-KV: relaunch stages with FV_WINDOW=1.
VALIDATED 2026-06-23: short prompt -> coherent gpt-oss-120B output across WA->WA->TX.
(destroyed earlier: the no-29600 set, flaky CA hosts 224600 + 392559)

## Ops
- Control: `cd phase0 && python3 fleet.py ls|eps|wait|exec|push|warm`
- Bootstrap: setup_box.sh (deps) + get_model.py (120b stages, +20b coord)
- Launch: `python3 launch_oss.py --stages tx1,tx2,ca1,wa1 --coord ca2 --max-ctx 98304 ...`
- Teardown when done: `vastai destroy instance <id>` (per box)
- PSK: ~/.shard_psk (gitignored)

## Progress log
- 2026-06-23: fleet rented. Provisioning + 120b download starting.
- 2026-06-23: RESULT — **28.18 tok/s decode at >100k context** (past the 20 target), greedy-exact,
  receipt `docs/receipts/gpt-oss-120b-100k-ngram-20260623.json`. Signed per-stage receipts
  demonstrated live (PROVE). All work committed + pushed (origin/master, leyten/anon).
- 2026-06-23: **fleet torn down** (all instances destroyed — no idle spend). Re-spin via the Launch
  command above (re-rent 3 distinct western cuda-13.2 hosts WITH `-p 29600:29600`; ~25min model dl).
