"""
telemetry.py — the structured logger (item 19) + the data contract.

Everything downstream reads THIS format: your math/analysis scripts (pandas),
and the future Three.js visualization. Locking the schema now means you never
have to reformat later.

Format: JSON Lines (.jsonl) — one JSON object per line, append-only, streamable.
Each line is one telemetry snapshot at a control interval:

    {
      "t":        float,   # sim time (s)
      "cwnd":     float,   # congestion window (packets)
      "inflight": float,   # bytes in flight (packets)
      "srtt":     float,   # smoothed RTT (s)
      "rtt_min":  float,   # min RTT seen (s)  -> latency floor
      "queue":    float,   # bottleneck queue occupancy (bytes)  -> viz fill level
      "goodput":  float,   # delivered throughput (bps)
      "loss":     float,   # cumulative loss fraction
      "reward":   float,   # reward this interval (null for non-RL runs)
      "action":   float,   # action taken (null for non-RL runs)
      "controller": str    # which algorithm produced this
    }
"""

from __future__ import annotations
import json


class TelemetryLogger:
    def __init__(self, path: str, controller_name: str):
        self.path = path
        self.controller_name = controller_name
        self._f = open(path, "w")
        self._n = 0

    def log(self, *, t, cwnd, inflight, srtt, rtt_min, queue,
            goodput, loss, reward=None, action=None):
        rec = {
            "t": round(t, 4),
            "cwnd": round(cwnd, 3),
            "inflight": round(inflight, 3),
            "srtt": round(srtt, 5),
            "rtt_min": round(rtt_min, 5) if rtt_min != float("inf") else 0.0,
            "queue": round(queue, 1),
            "goodput": round(goodput, 1),
            "loss": round(loss, 5),
            "reward": (round(reward, 4) if reward is not None else None),
            "action": (round(action, 4) if action is not None else None),
            "controller": self.controller_name,
        }
        self._f.write(json.dumps(rec) + "\n")
        self._n += 1

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def load_jsonl(path: str) -> list[dict]:
    """Read a telemetry file back (for analysis / viz)."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]