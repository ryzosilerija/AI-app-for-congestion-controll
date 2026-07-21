"""
game_net_client.py — receives ball positions over UDP and draws them.

This is the polished viewer, but instead of running physics locally, it draws
whatever the SERVER sends over the network. Run game_server.py first, then this.

    python game_net_client.py

When the network is smooth, balls move smoothly. Later, when we insert a
congested link, they will stutter here — that is the whole demo.
"""
from __future__ import annotations
import socket
import struct
import sys
import math
import pygame

ADDR = ("127.0.0.1", 9999)
WIDTH, HEIGHT = 800, 600
BG = (10, 12, 20)
TEXT_COLOR = (210, 220, 235)
TRAIL_FADE = 40


def unpack_snapshot(data):
    """Reverse of the server's pack_snapshot -> list of (id, x, y, r)."""
    (count,) = struct.unpack_from("!H", data, 0)
    off = 2
    out = []
    for _ in range(count):
        id, x, y, r = struct.unpack_from("!HhhB", data, off)
        off += 7
        out.append((id, x, y, r))
    return out


def speed_color_static(r):
    # no velocity on the client (we only get positions), color by size instead
    t = max(0.0, min(1.0, (r - 6) / 8))
    return (int(90 + t * 120), 170, int(255 - t * 120))


def draw_glow_ball(surf, x, y, r, color):
    glow = pygame.Surface((r * 6, r * 6), pygame.SRCALPHA)
    c = r * 3
    for rr, a in ((int(r * 2.6), 30), (int(r * 1.9), 45), (int(r * 1.3), 70)):
        pygame.draw.circle(glow, (*color, a), (c, c), rr)
    pygame.draw.circle(glow, color, (c, c), int(r))
    pygame.draw.circle(glow, (255, 255, 255, 90), (c, c), max(1, int(r * 0.4)))
    surf.blit(glow, (int(x - r * 3), int(y - r * 3)))


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(b"hello", ADDR)
    sock.setblocking(False)          # don't block the render loop waiting for packets

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Networked client (balls driven over UDP)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    trail = pygame.Surface((WIDTH, HEIGHT)); trail.fill(BG)

    bodies = []
    packets = 0
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit(); sys.exit()

        # drain all pending packets; keep the latest snapshot
        while True:
            try:
                data, _ = sock.recvfrom(65535)
                bodies = unpack_snapshot(data)
                packets += 1
            except BlockingIOError:
                break

        fade = pygame.Surface((WIDTH, HEIGHT)); fade.fill(BG); fade.set_alpha(TRAIL_FADE)
        trail.blit(fade, (0, 0))
        for (id, x, y, r) in bodies:
            draw_glow_ball(trail, x, y, r, speed_color_static(r))
        screen.blit(trail, (0, 0))

        hud = f"bodies: {len(bodies)}   packets: {packets}   (positions received over UDP)"
        screen.blit(font.render(hud, True, TEXT_COLOR), (10, 10))
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()