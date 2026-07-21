"""
game_client.py — Pygame viewer for the physics demo (LOCAL, no network yet).

This draws the PhysicsWorld directly in a window so you can SEE the bouncing
bodies and confirm the visual works before any networking exists. It also has
the interactive controls you wanted:

    UP / DOWN arrows : add / remove bodies  (scales network traffic later)
    LEFT / RIGHT     : slower / faster bodies (changes how latency-sensitive it is)
    SPACE            : pause
    ESC / close      : quit

Later, game_client will instead draw bodies received OVER THE NETWORK (through
your transport), and lag will make them stutter. For now it renders the local
sim at full speed as the "smooth reference" of how it should look.

Run:  python game_client.py
(Requires pygame:  pip install pygame)
"""

from __future__ import annotations
import sys
import pygame

from physics import PhysicsWorld

WIDTH, HEIGHT = 800, 600
FPS = 60
BG = (16, 18, 26)
BODY_COLOR = (90, 180, 255)
TEXT_COLOR = (200, 210, 225)


def scale_speed(world, factor):
    for o in world.bodies:
        o.vx *= factor
        o.vy *= factor


def add_bodies(world, n):
    import random
    rng = world.rng
    start = len(world.bodies)
    for i in range(n):
        r = rng.uniform(6, 14)
        from physics import Body
        world.bodies.append(Body(
            id=start + i,
            x=rng.uniform(r, world.width - r),
            y=rng.uniform(r, world.height - r),
            vx=rng.uniform(-180, 180),
            vy=rng.uniform(-180, 180),
            r=r,
        ))


def remove_bodies(world, n):
    for _ in range(min(n, len(world.bodies))):
        world.bodies.pop()


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Physics demo (local reference — no network)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    world = PhysicsWorld(WIDTH, HEIGHT, n_bodies=60, seed=1)
    paused = False

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_UP:
                    add_bodies(world, 20)
                elif e.key == pygame.K_DOWN:
                    remove_bodies(world, 20)
                elif e.key == pygame.K_LEFT:
                    scale_speed(world, 0.8)
                elif e.key == pygame.K_RIGHT:
                    scale_speed(world, 1.25)

        dt = 1.0 / FPS
        if not paused:
            world.step(dt)

        screen.fill(BG)
        for o in world.bodies:
            pygame.draw.circle(screen, BODY_COLOR, (int(o.x), int(o.y)), int(o.r))

        # HUD
        n = len(world.bodies)
        traffic = n * 8 * 30 * 8 / 1e6   # ~Mbps at 30Hz, 8 bytes/body
        lines = [
            f"bodies: {n}   (UP/DOWN to change)",
            f"~traffic at 30Hz: {traffic:.2f} Mbps   (LEFT/RIGHT = speed, SPACE = pause)",
        ]
        for i, t in enumerate(lines):
            screen.blit(font.render(t, True, TEXT_COLOR), (10, 10 + i * 20))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()