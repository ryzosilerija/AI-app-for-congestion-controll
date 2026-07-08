"""
demo_congestion.py — watch congestion happen.

This is your first "it's alive" check. It has NO protocol and NO AI — it just
blasts packets through the Channel at different send rates and shows what the
network does in response:

  - Send slower than the bottleneck  -> queue stays empty, nothing drops.
  - Send at the bottleneck rate      -> queue hovers, still fine.
  - Send faster than the bottleneck  -> queue fills, then packets DROP.

Seeing that transition is the proof your foundation works. Everything else
(ACKs, retransmits, the congestion controller, the AI) exists to *avoid* that
drop while keeping throughput high.
"""

from channel import Channel, Packet


def run_at_rate(send_rate_bps: float, duration_s: float = 1.0,
                packet_size: int = 1200, bottleneck_bps: float = 10e6):
    delivered = []

    ch = Channel(
        bandwidth_bps=bottleneck_bps,
        propagation_delay_s=0.02,
        buffer_bytes=30_000,   # ~25 packets of headroom
        loss_rate=0.0,
        seed=1,
    )
    ch.on_deliver = lambda pkt, t: delivered.append((pkt.pid, t))

    # Inter-packet gap to achieve the requested send rate.
    gap = (packet_size * 8) / send_rate_bps

    t = 0.0
    pid = 0
    # Feed packets in, advancing the channel clock as we go.
    while t < duration_s:
        ch.run_until(t)            # process any events up to now
        ch.send(Packet(pid, packet_size, t))
        pid += 1
        t += gap
    ch.run_until(duration_s + 1.0)  # drain tail

    s = ch.stats
    offered_mbps = send_rate_bps / 1e6
    goodput_mbps = (s["delivered"] * packet_size * 8) / 1e6 / duration_s
    drop_pct = 100 * (s["dropped_buffer"] + s["dropped_random"]) / max(s["sent"], 1)

    print(f"  offered={offered_mbps:5.1f} Mbps | "
          f"goodput={goodput_mbps:5.1f} Mbps | "
          f"sent={s['sent']:4d} delivered={s['delivered']:4d} "
          f"dropped={s['dropped_buffer']:4d} | "
          f"peak_queue={s['max_queue_bytes']:6d}B | "
          f"loss={drop_pct:4.1f}%")


if __name__ == "__main__":
    BOTTLENECK = 10e6  # 10 Mbps link
    print(f"Bottleneck link: {BOTTLENECK/1e6:.0f} Mbps\n")
    print("Sending BELOW, AT, and ABOVE the bottleneck rate:\n")
    for mult in (0.5, 0.9, 1.0, 1.5, 2.0, 4.0):
        run_at_rate(BOTTLENECK * mult, bottleneck_bps=BOTTLENECK)
    print("\nNotice: once offered load exceeds the bottleneck, the queue")
    print("saturates and packets start dropping. That's congestion. A good")
    print("controller pushes goodput close to the bottleneck WITHOUT the loss.")