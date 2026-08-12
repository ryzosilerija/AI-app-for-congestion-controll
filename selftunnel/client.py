"""Client — runs on YOUR machine, next to the game.

Holds an outbound transport connection to the relay (defeating NAT: you dial
out), and forwards tunnel traffic to/from your local game port.

    python client.py --relay <relay-ip> --mode tcp --local-port 25565 --cc aimd
    python client.py --relay <relay-ip> --mode udp --local-port 27015 --cc aimd
"""
import argparse
import socket
import threading

from transport import Transport
from congestion import make_controller


def connect(relay_ip, ctrl_port, cc) -> Transport:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(b"HELLO", (relay_ip, ctrl_port))       # announce ourselves
    print(f"[client] dialed relay {relay_ip}:{ctrl_port}")
    return Transport(sock, (relay_ip, ctrl_port), cc)


def run_tcp(local_port, tunnel: Transport):
    # tunnel -> local game (open a fresh local connection), and back
    game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    game.connect(("127.0.0.1", local_port))
    print(f"[client] forwarding to local tcp/{local_port}")

    def up():                        # game -> tunnel
        try:
            while True:
                data = game.recv(4096)
                if not data:
                    break
                tunnel.send_stream(data)
        except OSError:
            pass
    threading.Thread(target=up, daemon=True).start()

    while True:                      # tunnel -> game
        data = tunnel.recv_stream(timeout=1.0)
        if data:
            game.sendall(data)


def run_udp(local_port, tunnel: Transport):
    game = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = ("127.0.0.1", local_port)
    print(f"[client] forwarding to local udp/{local_port}")

    def up():                        # game -> tunnel
        while True:
            data = tunnel.recv_stream(timeout=1.0)
            if data:
                game.sendto(data, target)
    threading.Thread(target=up, daemon=True).start()

    while True:                      # local replies -> tunnel
        try:
            data, _ = game.recvfrom(2048)
            tunnel.send_stream(data)
        except OSError:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relay", required=True, help="relay public IP")
    ap.add_argument("--mode", choices=["tcp", "udp"], required=True)
    ap.add_argument("--local-port", type=int, required=True,
                    help="the game's port on this machine")
    ap.add_argument("--ctrl-port", type=int, default=9000)
    ap.add_argument("--cc", default="aimd")
    args = ap.parse_args()

    cc = make_controller(args.cc)
    tunnel = connect(args.relay, args.ctrl_port, cc)
    if args.mode == "tcp":
        run_tcp(args.local_port, tunnel)
    else:
        run_udp(args.local_port, tunnel)


if __name__ == "__main__":
    main()
