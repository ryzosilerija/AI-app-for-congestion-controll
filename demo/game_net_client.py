"""
game_net_client.py — receives physics over UDPLink (sending ACKs back so the
server can measure RTT), draws the balls, and shows a live network telemetry HUD.

    python game_net_client.py   (run game_server.py first)

The HUD shows RTT / loss / send-rate reported by the server, plus the client's
own packet count and measured inter-packet gap (jitter you can SEE as stutter).
"""
from __future__ import annotations
import socket
import struct
import sys
import time
import pygame

from udp_link import UDPLink

ADDR = ("127.0.0.1", 9999)
WIDTH, HEIGHT = 800, 600
BG = (10, 12, 20)
TEXT = (210, 220, 235)
DIM = (120, 130, 150)
TRAIL_FADE = 40


def unpack_snapshot(data):
    (count,) = struct.unpack_from("!H", data, 0)
    off = 2
    bodies = []
    for _ in range(count):
        id, x, y, r = struct.unpack_from("!HhhB", data, off)
        off += 7
        bodies.append((id, x, y, r))
    srtt, loss, rate = struct.unpack_from("!fff", data, off)
    off += 12
    (nlen,) = struct.unpack_from("!B", data, off); off += 1
    ctrl = data[off:off+nlen].decode(errors="replace")
    return bodies, (srtt, loss, rate, ctrl)


def color_by_size(r):
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
    sock.sendto(b"hello", ADDR)          # register with server
    link = UDPLink(sock, peer_addr=ADDR)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Networked client + telemetry")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 15)
    big = pygame.font.SysFont("consolas", 18, bold=True)
    trail = pygame.Surface((WIDTH, HEIGHT)); trail.fill(BG)

    state = {"bodies": [], "srtt": 0.0, "loss": 0.0, "rate": 0.0, "ctrl": "-"}
    packets = 0
    last_pkt_t = time.monotonic()
    gap_ms = 0.0

    def on_data(payload):
        nonlocal packets, last_pkt_t, gap_ms
        bodies, (srtt, loss, rate, ctrl) = unpack_snapshot(payload)
        state["bodies"] = bodies
        state["srtt"] = srtt; state["loss"] = loss; state["rate"] = rate
        state["ctrl"] = ctrl
        now = time.monotonic()
        gap_ms = (now - last_pkt_t) * 1000
        last_pkt_t = now
        packets += 1

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit(); sys.exit()
            elif e.type == pygame.KEYDOWN:
                # send control commands to the server (CMD codes match server)
                if e.key == pygame.K_UP:
                    link.send_cmd(1, 20)      # add balls
                elif e.key == pygame.K_DOWN:
                    link.send_cmd(2, 20)      # remove balls
                elif e.key == pygame.K_RIGHT:
                    link.send_cmd(3, 0)       # faster
                elif e.key == pygame.K_LEFT:
                    link.send_cmd(4, 0)       # slower
                elif e.key == pygame.K_LEFTBRACKET:
                    link.send_cmd(5, 0)       # tighten bottleneck
                elif e.key == pygame.K_RIGHTBRACKET:
                    link.send_cmd(6, 0)       # loosen bottleneck
                elif e.key in (pygame.K_1, pygame.K_KP1):
                    link.send_cmd(7, 0)       # controller: blind
                elif e.key in (pygame.K_2, pygame.K_KP2, 1073741914):
                    link.send_cmd(8, 0)       # controller: cubic
                elif e.key in (pygame.K_3, pygame.K_KP3, 1073741915):
                    link.send_cmd(9, 0)       # controller: delay-CC stand-in
                elif e.key in (pygame.K_4, pygame.K_KP4, 1073741916):
                    link.send_cmd(10, 0)      # controller: REAL trained agent
                elif e.key == pygame.K_r:
                    link.send_cmd(11, 0)      # reset ball speed to normal

        link.poll(on_data=on_data)       # receives data, auto-sends ACKs

        fade = pygame.Surface((WIDTH, HEIGHT)); fade.fill(BG); fade.set_alpha(TRAIL_FADE)
        trail.blit(fade, (0, 0))
        for (id, x, y, r) in state["bodies"]:
            draw_glow_ball(trail, x, y, r, color_by_size(r))
        screen.blit(trail, (0, 0))

        # --- BIG CONTROLLER BANNER --- unmissable at the top
        ctrl = state.get("ctrl", "-")
        if "CUBIC" in ctrl:
            banner_col = (255, 80, 60)      # red = loss-based (bufferbloat)
        elif "PPO" in ctrl or "AGENT" in ctrl:
            banner_col = (255, 210, 0)      # gold = real trained agent
        elif "DELAY" in ctrl:
            banner_col = (60, 220, 100)     # green = delay-CC stand-in
        else:
            banner_col = (160, 160, 160)    # grey = blind
        banner = pygame.Surface((WIDTH, 44))
        banner.fill((20, 20, 28))
        label = big.render(f"▶  {ctrl}  ◀", True, banner_col)
        banner.blit(label, (WIDTH//2 - label.get_width()//2, 6))
        screen.blit(banner, (0, 0))

        # telemetry HUD
        rows = [
            ("NETWORK TELEMETRY", TEXT, font),
            (f"balls          : {len(state['bodies'])}", TEXT, font),
            (f"RTT (smoothed) : {state['srtt']:6.1f} ms", TEXT, font),
            (f"loss           : {state['loss']:6.1f} %", TEXT, font),
            (f"send rate      : {state['rate']:6.2f} Mbps", TEXT, font),
            (f"packets rcvd   : {packets}", DIM, font),
            (f"inter-pkt gap  : {gap_ms:6.1f} ms  (jitter = stutter)", DIM, font),
            ("arrows: UP/DOWN=balls  LEFT/RIGHT=speed  R=reset speed", DIM, font),
            ("numpad: 1=blind  2=cubic  3=delay-CC  4=PPO AGENT (real)", DIM, font),
        ]
        y = 52   # below the banner
        for text, col, fnt in rows:
            screen.blit(fnt.render(text, True, col), (10, y))
            y += 20

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()