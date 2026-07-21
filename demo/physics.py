"""
physics.py — a simple 2D physics sandbox for the congestion-control demo.

N circular bodies bounce around a box and collide elastically. This is the
SERVER-SIDE state that will be streamed to clients over your transport. It is
deliberately self-contained and network-free so it can be tested on its own
before any networking exists.

The demo value: chaotic collisions make network lag visually obvious — under
congestion the bodies stutter and jump; with good congestion control they move
smoothly. Scale N up to increase the network traffic (more bodies = more
position updates per tick) until it congests a throttled link.

Run standalone (no graphics):  python physics.py    -> prints state, self-tests.
"""

from __future__ import annotations
import math
import random


class Body:
    __slots__ = ("id", "x", "y", "vx", "vy", "r", "m")

    def __init__(self, id, x, y, vx, vy, r):
        self.id = id
        self.x = x; self.y = y
        self.vx = vx; self.vy = vy
        self.r = r
        self.m = r * r          # mass proportional to area (r^2)


class PhysicsWorld:
    """Server-side authoritative physics. Fixed timestep for determinism."""

    def __init__(self, width=800, height=600, n_bodies=50, seed=1):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.bodies = []
        for i in range(n_bodies):
            r = self.rng.uniform(6, 14)
            self.bodies.append(Body(
                id=i,
                x=self.rng.uniform(r, width - r),
                y=self.rng.uniform(r, height - r),
                vx=self.rng.uniform(-180, 180),
                vy=self.rng.uniform(-180, 180),
                r=r,
            ))

    def step(self, dt):
        """Advance the simulation by dt seconds."""
        b = self.bodies
        # integrate positions
        for o in b:
            o.x += o.vx * dt
            o.y += o.vy * dt
            # wall bounces
            if o.x - o.r < 0:
                o.x = o.r; o.vx = abs(o.vx)
            elif o.x + o.r > self.width:
                o.x = self.width - o.r; o.vx = -abs(o.vx)
            if o.y - o.r < 0:
                o.y = o.r; o.vy = abs(o.vy)
            elif o.y + o.r > self.height:
                o.y = self.height - o.r; o.vy = -abs(o.vy)

        # pairwise elastic collisions (O(n^2) — fine for a few hundred bodies)
        n = len(b)
        for i in range(n):
            a = b[i]
            for j in range(i + 1, n):
                c = b[j]
                dx = c.x - a.x; dy = c.y - a.y
                dist2 = dx * dx + dy * dy
                rsum = a.r + c.r
                if dist2 < rsum * rsum and dist2 > 1e-9:
                    dist = math.sqrt(dist2)
                    # normal
                    nx = dx / dist; ny = dy / dist
                    # separate overlap
                    overlap = rsum - dist
                    total_m = a.m + c.m
                    a.x -= nx * overlap * (c.m / total_m)
                    a.y -= ny * overlap * (c.m / total_m)
                    c.x += nx * overlap * (a.m / total_m)
                    c.y += ny * overlap * (a.m / total_m)
                    # relative velocity along normal
                    dvx = c.vx - a.vx; dvy = c.vy - a.vy
                    vn = dvx * nx + dvy * ny
                    if vn < 0:   # approaching
                        # 1D elastic impulse along normal
                        imp = (2 * vn) / total_m
                        a.vx += imp * c.m * nx
                        a.vy += imp * c.m * ny
                        c.vx -= imp * a.m * nx
                        c.vy -= imp * a.m * ny

    def snapshot(self):
        """Return the state to stream to clients: list of (id, x, y, r)."""
        return [(o.id, round(o.x, 1), round(o.y, 1), round(o.r, 1)) for o in self.bodies]

    def energy(self):
        """Total kinetic energy — used as a self-test (should stay ~constant)."""
        return sum(0.5 * o.m * (o.vx * o.vx + o.vy * o.vy) for o in self.bodies)


if __name__ == "__main__":
    # Self-test: run the sim and confirm bodies stay in bounds and energy is
    # roughly conserved (elastic collisions shouldn't create or destroy much).
    world = PhysicsWorld(n_bodies=40, seed=1)
    e0 = world.energy()
    dt = 1 / 60
    in_bounds = True
    for _ in range(600):   # 10 seconds at 60fps
        world.step(dt)
        for o in world.bodies:
            if not (0 <= o.x <= world.width and 0 <= o.y <= world.height):
                in_bounds = False
    e1 = world.energy()
    print(f"bodies: {len(world.bodies)}")
    print(f"all in bounds after 10s: {in_bounds}")
    print(f"energy start {e0:.0f}, end {e1:.0f}, ratio {e1/e0:.3f} (should be ~1.0)")
    snap = world.snapshot()
    print(f"snapshot sample (first 3 of {len(snap)}): {snap[:3]}")
    # bytes-per-snapshot estimate for traffic planning
    import sys
    approx_bytes = len(snap) * 8   # ~8 bytes packed per body (id,x,y as int16-ish)
    for n in (50, 200, 1000):
        bps = n * 8 * 30 * 8   # bytes*ticks/s*bits — 30 ticks/s
        print(f"  {n} bodies @30Hz ~= {bps/1e6:.2f} Mbps of position traffic")