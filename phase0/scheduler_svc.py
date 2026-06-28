"""scheduler_svc — thin HTTP wrapper over shard/scheduler.py for the c0mpute orchestrator.

The orchestrator (TypeScript) owns assembly + payments but must not re-implement the
VRAM-fit + min-latency ring solver (it's tested Python in shard/scheduler.py). This service
is the one network call the orchestrator makes per ring job: hand it the joined workers, get
back each stage's contiguous layer block + the ring order + the coordinator pick.

Boundary law (docs/INTEGRATION.md): this knows layers, vram, rtt — nothing about $ZERO,
accounts, or receipts. Pure control-plane math. Stdlib only (no deps): runs anywhere the
orchestrator box can reach.

  python3 scheduler_svc.py --port 8088

  POST /plan
  {
    "model": "GLM-5.2",
    "total_layers": 78,
    "gb_per_layer": 1.05,           # model bytes/layer (caller knows the quant)
    "kv_gb_per_layer": 0.04,        # KV bytes/layer at target ctx (optional)
    "headroom_gb": 2.0,             # optional
    "boundary_gb": 1.0,             # optional (embed+lm_head slack)
    "coordinator": "<node_id>",     # optional: pin the entry node; else best-uplink heuristic
    "nodes": [
      {"node_id": "A", "vram_gb": 48, "rtt_ms": {"B": 30, "C": 40}},
      {"node_id": "B", "vram_gb": 24, "rtt_ms": {"A": 30, "C": 25}},
      {"node_id": "C", "vram_gb": 24, "rtt_ms": {"A": 40, "B": 25}}
    ]
  }

  -> 200
  {
    "ok": true,
    "coordinator": "A",
    "ring_order": ["A", "B", "C"],          # head-first stage order (coordinator drives from head)
    "stages": [
      {"stage": 0, "node_id": "A", "lo": 0,  "hi": 30, "n_layers": 30},
      {"stage": 1, "node_id": "B", "lo": 30, "hi": 54, "n_layers": 24},
      {"stage": 2, "node_id": "C", "lo": 54, "hi": 78, "n_layers": 24}
    ]
  }

  -> 400 {"ok": false, "error": "insufficient VRAM: ..."} when the pool can't hold the model.
"""

import argparse, json, sys, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# import the real solver — service lives in phase0/, scheduler in shard/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shard.scheduler import Scheduler, JoinedNode


def plan(req: dict) -> dict:
    """pure function: req dict -> plan dict. raises ValueError on infeasible fit.

    factored out of the HTTP layer so the offline test drives it with no socket.
    """
    model = req["model"]
    total_layers = int(req["total_layers"])
    gb_per_layer = float(req["gb_per_layer"])
    kv_gb = float(req.get("kv_gb_per_layer", 0.0))
    headroom = float(req.get("headroom_gb", 2.0))
    boundary = float(req.get("boundary_gb", 1.0))
    nodes = req["nodes"]
    if not nodes:
        raise ValueError("no nodes")

    sch = Scheduler(model, total_layers)
    for n in nodes:
        sch.register(JoinedNode(node_id=n["node_id"], vram_gb=float(n["vram_gb"]),
                                rtt_ms={k: float(v) for k, v in (n.get("rtt_ms") or {}).items()}))

    # coordinator: caller pin, else the node with the lowest mean rtt to the rest (best-connected
    # entry/return depot — the coordinator pays the entry hop out and the direct-return hop back
    # every round, so a well-connected depot matters more than its vram).
    coord = req.get("coordinator")
    if not coord:
        def mean_rtt(nid):
            r = sch.nodes[nid].rtt_ms
            return sum(r.values()) / len(r) if r else float("inf")
        coord = min(sch.nodes, key=mean_rtt)
    if coord not in sch.nodes:
        raise ValueError(f"coordinator {coord!r} not in node set")

    alloc = sch.allocate(gb_per_layer, kv_gb, headroom, boundary)   # node_id -> LayerRange
    # ring_order: the coordinator is CO-LOCATED on the head stage (it serves layers AND drives,
    # exactly as phase0/launch_libp2p.py runs it: coordinator on the head box, --next to the
    # local head engine). So the served ring is [coord-as-head] + the min-latency loop through
    # the remaining stages. topology() returns the NON-coordinator order; we prepend the head.
    ring = [coord] + sch.topology(coord)

    # map the ring order onto the allocated blocks. allocate() hands out blocks fat-node-first
    # (contiguous [lo,hi) covering [0:total]); we re-walk them in RING order so stage k's block
    # is contiguous along the wire path. re-tile lo/hi in ring order to keep blocks contiguous
    # per the engine's --lo/--hi contract (each stage reindexes 0-based locally).
    counts = {nid: (lr.end - lr.start) for nid, lr in alloc.items()}
    stages, cur = [], 0
    for k, nid in enumerate(ring):
        c = counts.get(nid, 0)
        if c == 0:
            continue                      # a node the fit gave 0 layers (tiny vram) is not a stage
        stages.append({"stage": len(stages), "node_id": nid, "lo": cur, "hi": cur + c, "n_layers": c})
        cur += c
    if cur != total_layers:
        raise ValueError(f"coverage gap: tiled {cur} layers != model {total_layers}")

    return {
        "ok": True,
        "model": model,
        "coordinator": coord,
        "ring_order": [s["node_id"] for s in stages],
        "stages": stages,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        b = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        if self.path != "/plan":
            self._send(404, {"ok": False, "error": "POST /plan"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, {"ok": False, "error": f"bad json: {e}"})
            return
        try:
            self._send(200, plan(req))
        except ValueError as e:
            self._send(400, {"ok": False, "error": str(e)})
        except KeyError as e:
            self._send(400, {"ok": False, "error": f"missing field: {e}"})
        except Exception as e:                          # never leak a stack to the orchestrator
            self._send(500, {"ok": False, "error": f"internal: {e}"})

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True})
        else:
            self._send(404, {"ok": False, "error": "GET /health"})

    def log_message(self, format, *args):               # quiet; the orchestrator logs its own calls
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"scheduler_svc on {a.host}:{a.port} (POST /plan, GET /health)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
