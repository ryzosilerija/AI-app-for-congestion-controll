"""Relay — runs on a machine with a PUBLIC IP.

It does two jobs:
  1. Accepts one tunnel client (your machine) over the UDP transport.
  2. Opens a PUBLIC listener (TCP or UDP) that your friend connects to, and
     shuttles that traffic through the transport to the client.

On startup it prints the exact link to share: <public-ip>:<public-port>.

    python relay.py --mode tcp  --public-port 25565 --cc aimd
    python relay.py --mode udp  --public-port 27015 --cc aimd

Control port (where the client connects in) defaults to udp/9000.
"""
import argparse
import socket
import threading
import urllib.request

from transport import Transport
from congestion import make_controller


def public_ip() -> str:
    try:
        return urllib.request.urlopen("https://api.ipify.org", timeout=5).read().decode()
    except Exception:
        # fall back to the primary outbound interface address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()


def wait_for_client(ctrl_port, cc) -> Transport:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", ctrl_port))
    print(f"[relay] waiting for tunnel client on udp/{ctrl_port} ...")
    # first datagram tells us the client's address; hand the socket to Transport
    data, addr = sock.recvfrom(2048)
    print(f"[relay] client connected from {addr[0]}:{addr[1]}")
    return Transport(sock, addr, cc)


def serve_tcp(public_port, tunnel: Transport):
    ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("0.0.0.0", public_port))
    ls.listen(8)
    print(f"[relay] public TCP listener on :{public_port}")
    while True:
        conn, addr = ls.accept()
        print(f"[relay] friend connected (tcp) from {addr[0]}:{addr[1]}")
        threading.Thread(target=_pump_tcp, args=(conn, tunnel), daemon=True).start()


def _pump_tcp(conn, tunnel):
    # friend -> tunnel
    def up():
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                tunnel.send_stream(data)
        except OSError:
            pass
    threading.Thread(target=up, daemon=True).start()
    # tunnel -> friend
    try:
        while True:
            data = tunnel.recv_stream(timeout=1.0)
            if data:
                conn.sendall(data)
    except OSError:
        pass


def serve_udp(public_port, tunnel: Transport):
    us = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    us.bind(("0.0.0.0", public_port))
    print(f"[relay] public UDP listener on :{public_port}")
    friend = {"addr": None}

    def down():                      # tunnel -> friend
        while True:
            data = tunnel.recv_stream(timeout=1.0)
            if data and friend["addr"]:
                us.sendto(data, friend["addr"])
    threading.Thread(target=down, daemon=True).start()

    while True:                      # friend -> tunnel
        data, addr = us.recvfrom(2048)
        friend["addr"] = addr
        tunnel.send_stream(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tcp", "udp"], required=True)
    ap.add_argument("--public-port", type=int, required=True)
    ap.add_argument("--ctrl-port", type=int, default=9000)
    ap.add_argument("--cc", default="aimd", help="congestion controller name")
    args = ap.parse_args()

    cc = make_controller(args.cc)
    ip = public_ip()

    print("=" * 52)
    print(f"  SHARE THIS LINK WITH YOUR FRIEND:")
    print(f"     {ip}:{args.public_port}   ({args.mode.upper()})")
    print(f"  congestion control: {args.cc}")
    print("=" * 52)

    tunnel = wait_for_client(args.ctrl_port, cc)
    if args.mode == "tcp":
        serve_tcp(args.public_port, tunnel)
    else:
        serve_udp(args.public_port, tunnel)


if __name__ == "__main__":
    main()
