"""
game_server.py — physics streamed through UDPLink with WINDOW-BASED flow control.

The controller sets a cwnd (congestion window in bytes). The server only sends
when bytes_inflight < cwnd. This gives the controller real leverage over latency:
shrink the window -> stop sending -> queue drains -> RTT drops. That's the
mechanism that makes CUBIC bloat (big window, fills buffer) while delay-CC
keeps latency low (small window when RTT is high).

    python game_server.py
"""
from __future__ import annotations
import socket
import struct
import time

from physics import PhysicsWorld, Body
from udp_link import UDPLink
from bottleneck import Bottleneck
import demo_controllers
import agent_controller as _agentmod

ADDR = ("127.0.0.1", 9999)
TICK_HZ = 30

CMD_ADD = 1
CMD_REMOVE = 2
CMD_FASTER = 3
CMD_SLOWER = 4
CMD_TIGHTEN = 5
CMD_LOOSEN = 6
CMD_CTRL_BLIND = 7
CMD_CTRL_CUBIC = 8
CMD_CTRL_DELAY = 9
CMD_CTRL_AGENT = 10
CMD_RESET_SPEED = 11


_LOG = []


def save_report(log, path="demo_report.csv"):
    import csv
    if not log:
        print("server: no data to save")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader(); w.writerows(log)
    print(f"server: saved {len(log)} rows to {path}")


def pack_snapshot(bodies, stats):
    parts = [struct.pack("!H", len(bodies))]
    for o in bodies:
        parts.append(struct.pack("!HhhB", o.id & 0xFFFF, int(o.x), int(o.y),
                                 min(255, int(o.r))))
    parts.append(struct.pack("!fff", stats["srtt_ms"], stats["loss_pct"],
                             stats["rate_mbps"]))
    name = stats.get("ctrl", "").encode()[:40]
    parts.append(struct.pack("!B", len(name)) + name)
    return b"".join(parts)


def main():
    global _LOG
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(ADDR)
    print(f"server: bound to {ADDR}, waiting for client...")
    data, client_addr = sock.recvfrom(1024)
    print(f"server: client connected from {client_addr}")

    link = UDPLink(sock, peer_addr=client_addr)
    world = PhysicsWorld(800, 600, n_bodies=60, seed=1)
    bottleneck = Bottleneck(rate_mbps=2.0, base_delay_ms=20.0, queue_bytes=30_000)
    raw_send = sock.sendto

    # route link's outgoing packets through the bottleneck
    def bottlenecked_send(data, dest):
        bottleneck.offer(data, dest)
    link.sock_sendto_override = bottlenecked_send

    # load the real trained agent
    _real_agent = _agentmod.load_agent()
    if _real_agent is not None:
        controller = _agentmod.AgentCC(_real_agent)
        print(f'server: using REAL trained agent as default controller')
    else:
        controller = demo_controllers.make('delay')
        print('server: agent.pt not found, using delay-CC stand-in')

    log = _LOG
    next_id = [len(world.bodies)]

    def add_balls(n):
        rng = world.rng
        for _ in range(n):
            r = rng.uniform(6, 14)
            world.bodies.append(Body(
                id=next_id[0], x=rng.uniform(r, world.width - r),
                y=rng.uniform(r, world.height - r),
                vx=rng.uniform(-180, 180), vy=rng.uniform(-180, 180), r=r))
            next_id[0] += 1

    def _set_controller(name):
        nonlocal controller
        if name == 'agent':
            if _real_agent is not None:
                controller = _agentmod.AgentCC(_real_agent)
            else:
                controller = demo_controllers.make('delay')
        else:
            controller = demo_controllers.make(name)
        print(f'server: controller -> {controller.name}')

    def on_cmd(cmd, arg):
        try:
            if cmd == CMD_ADD:
                add_balls(arg or 20)
            elif cmd == CMD_REMOVE:
                for _ in range(min(arg or 20, len(world.bodies))):
                    world.bodies.pop()
            elif cmd == CMD_FASTER:
                for o in world.bodies:
                    o.vx *= 1.25; o.vy *= 1.25
            elif cmd == CMD_SLOWER:
                for o in world.bodies:
                    o.vx *= 0.8; o.vy *= 0.8
            elif cmd == CMD_TIGHTEN:
                bottleneck.rate_bps = max(0.01e6, bottleneck.rate_bps * 0.7)
            elif cmd == CMD_LOOSEN:
                bottleneck.rate_bps = bottleneck.rate_bps * 1.4
            elif cmd == CMD_CTRL_BLIND:
                _set_controller('blind')
            elif cmd == CMD_CTRL_CUBIC:
                _set_controller('cubic')
            elif cmd == CMD_CTRL_DELAY:
                _set_controller('delay')
            elif cmd == CMD_CTRL_AGENT:
                _set_controller('agent')
            elif cmd == CMD_RESET_SPEED:
                import math
                for o in world.bodies:
                    speed = math.hypot(o.vx, o.vy)
                    if speed > 0:
                        o.vx = o.vx / speed * 150
                        o.vy = o.vy / speed * 150
                print("server: speed reset")
            print(f"server: cmd {cmd} -> {len(world.bodies)} bodies")
        except Exception as e:
            print(f"server: ERROR in on_cmd {cmd}: {e}")

    dt = 1.0 / TICK_HZ
    next_t = time.monotonic()
    bytes_window = 0
    rate_t0 = time.monotonic()
    rate_mbps = 0.0
    last_acked = 0

    while True:
        try:
            link.poll(on_cmd=on_cmd)
            bottleneck.drain(raw_send)
            world.step(dt)
        except Exception as e:
            print(f"server: loop error: {e}")
            continue

        # --- update the controller with real feedback ---
        st = link.stats()
        loss_events = link.pop_loss_events()
        new_acked = st['acked'] - last_acked
        bytes_acked = new_acked * 500   # approximate bytes per ack
        last_acked = st['acked']

        controller.update(st['srtt_ms'] / 1000.0, st['rtt_min_ms'] / 1000.0,
                          loss_events, bytes_acked,
                          bytes_inflight=link.bytes_inflight)

        # --- WINDOW-BASED GATING: only send if inflight < cwnd ---
        st['rate_mbps'] = rate_mbps
        st['ctrl'] = controller.name
        payload = pack_snapshot(world.bodies, st)

        if link.bytes_inflight < controller.cwnd:
            link.send_data(payload)
            bytes_window += len(payload)
        # else: window full, skip this tick's send (queue drains via ACKs)

        # log + print once per second
        now = time.monotonic()
        if now - rate_t0 >= 1.0:
            rate_mbps = bytes_window * 8 / 1e6 / (now - rate_t0)
            bytes_window = 0
            rate_t0 = now
            log.append({
                'controller': controller.name,
                'rtt_ms': round(st['srtt_ms'], 1),
                'loss_pct': round(st['loss_pct'], 2),
                'rate_mbps': round(rate_mbps, 3),
                'cwnd': controller.cwnd,
                'inflight': link.bytes_inflight,
                'bodies': len(world.bodies),
                'queue_bytes': bottleneck.queue_bytes,
                'dropped': bottleneck.dropped,
            })
            print(f"server: [{controller.name[:10]:10s}] RTT {st['srtt_ms']:6.1f}ms  "
                  f"cwnd {controller.cwnd:6d}  inflight {link.bytes_inflight:6d}  "
                  f"rate {rate_mbps:.2f}Mbps  balls {len(world.bodies)}")

        next_t += dt
        sleep = next_t - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)

    return log


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nserver: shutting down...")
    save_report(_LOG)