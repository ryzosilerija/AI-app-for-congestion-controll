"""
simulate.py — wires everything together and runs the clock (the experiment runner, item 17 seed).

Topology:
    Sender --(forward Channel)--> Receiver
    Sender <--(reverse Channel)-- Receiver   (ACKs)

We step a shared virtual clock in small ticks. Each tick we:
  1. deliver any packets/acks whose time has come (the channels' events),
  2. let the sender pump new packets (within cwnd),
  3. check retransmission timeouts,
  4. record telemetry.

This is the loop the RL agent will later plug into (it just reads the sender's
state and sets the controller's window each control interval instead of AIMD).
"""

from __future__ import annotations
from channel import Channel
from transport import Sender, Receiver, DataPacket, AckPacket
from controllers import AIMD, FixedWindow


def run(controller, *, bottleneck_bps=10e6, delay_s=0.02, buffer_bytes=60_000,
        loss_rate=0.0, total_bytes=4_000_000, tick=0.001, max_time=30.0,
        mss=1200, seed=1):
    forward = Channel(bottleneck_bps, delay_s, buffer_bytes, loss_rate, seed)
    reverse = Channel(bottleneck_bps, delay_s, buffer_bytes, 0.0, seed + 1)

    sender = Sender(controller, forward, reverse, mss=mss, total_bytes=total_bytes)
    receiver = Receiver(reverse, mss=mss)

    forward.on_deliver = lambda pkt, t: receiver.on_data_received(pkt, t)
    reverse.on_deliver = lambda pkt, t: sender.on_ack_received(pkt, t)

    now = 0.0
    next_record = 0.0
    sender.pump(now)

    while now < max_time and not sender.finished:
        now += tick
        forward.run_until(now)
        reverse.run_until(now)
        sender.check_timeouts(now)
        sender.pump(now)
        if now >= next_record:
            sender.record(now)
            next_record += 0.02   # 50 Hz telemetry

    fwd = forward.stats
    completion = now
    goodput_mbps = (sender.acked_bytes * 8) / 1e6 / completion
    loss_pct = 100 * (fwd["dropped_buffer"] + fwd["dropped_random"]) / max(fwd["sent"], 1)
    return {
        "controller": type(controller).__name__,
        "completion_s": completion,
        "goodput_mbps": goodput_mbps,
        "loss_pct": loss_pct,
        "final_cwnd_pkts": controller.cwnd_packets,
        "srtt_ms": (sender.rtt.srtt or 0) * 1000,
        "rtt_min_ms": (sender.rtt.rtt_min if sender.rtt.rtt_min != float("inf") else 0) * 1000,
        "trace": sender.trace,
    }


def sawtooth_ascii(trace, width=64, height=14):
    """Tiny terminal plot of cwnd over time so you can SEE the sawtooth."""
    if not trace:
        return "(no data)"
    times = [r[0] for r in trace]
    cwnds = [r[1] for r in trace]
    tmax, cmax = max(times), max(cwnds) or 1
    grid = [[" "] * width for _ in range(height)]
    for t, c in zip(times, cwnds):
        x = min(width - 1, int(t / tmax * (width - 1)))
        y = min(height - 1, int(c / cmax * (height - 1)))
        grid[height - 1 - y][x] = "*"
    lines = ["".join(row) for row in grid]
    out = [f"cwnd (max {cmax:.0f} pkts)"]
    out += [f"{l}" for l in lines]
    out.append("0" + " " * (width - 6) + f"{tmax:.1f}s")
    return "\n".join(out)


if __name__ == "__main__":
    print("=" * 70)
    print("AIMD vs FixedWindow on a 10 Mbps / 20ms link (0.1% random loss)")
    print("=" * 70)

    for ctrl in (FixedWindow(init_cwnd_packets=20),
                 AIMD(init_cwnd_packets=2)):
        r = run(ctrl, loss_rate=0.001)
        print(f"\n{r['controller']}:")
        print(f"  completion : {r['completion_s']:.2f} s")
        print(f"  goodput    : {r['goodput_mbps']:.2f} Mbps "
              f"(link is 10 Mbps)")
        print(f"  loss       : {r['loss_pct']:.2f} %")
        print(f"  final cwnd : {r['final_cwnd_pkts']:.1f} pkts")
        print(f"  SRTT/min   : {r['srtt_ms']:.1f} / {r['rtt_min_ms']:.1f} ms "
              f"(min ~= 40ms RTT floor)")

    print("\n" + "-" * 70)
    print("AIMD congestion window over time — the classic sawtooth:")
    print("-" * 70)
    r = run(AIMD(init_cwnd_packets=2), loss_rate=0.001)
    print(sawtooth_ascii(r["trace"]))