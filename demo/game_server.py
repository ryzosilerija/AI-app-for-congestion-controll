"""
game_server.py — runs the physics simulation and streams ball positions to a
client over real UDP. (Step 2: physics over the network, still using RAW udp —
your transport/controller get inserted in the NEXT step.)

    python game_server.py

Waits for a client hello, then sends a snapshot of all balls ~30x/second.
Wire format per snapshot: [count:uint16][ id:uint16, x:int16, y:int16, r:uint8 ] * count
"""
from __future__ import annotations
import socket
import struct
import time

from physics import PhysicsWorld

ADDR = ("127.0.0.1", 9999)
TICK_HZ = 30
N_BODIES = 60


def pack_snapshot(bodies):
    """Pack the ball list into bytes. count, then 7 bytes per body."""
    parts = [struct.pack("!H", len(bodies))]
    for o in bodies:
        parts.append(struct.pack("!HhhB", o.id & 0xFFFF,
                                 int(o.x), int(o.y), min(255, int(o.r))))
    return b"".join(parts)


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(ADDR)
    print(f"server: bound to {ADDR}, waiting for client...")
    _, client_addr = sock.recvfrom(1024)
    print(f"server: client connected from {client_addr}")

    world = PhysicsWorld(800, 600, n_bodies=N_BODIES, seed=1)
    dt = 1.0 / TICK_HZ
    next_t = time.monotonic()
    while True:
        world.step(dt)
        sock.sendto(pack_snapshot(world.bodies), client_addr)
        next_t += dt
        sleep = next_t - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)


if __name__ == "__main__":
    main()