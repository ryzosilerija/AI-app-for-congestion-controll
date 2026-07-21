"""
udp_test.py — the smallest real-network step: two real UDP sockets exchanging
bytes on localhost. Proves real networking works BEFORE building transport on top.

    Terminal 1:   python udp_test.py server
    Terminal 2:   python udp_test.py client

Server sends a counter every 100ms; client prints what it receives. Increasing
numbers on the client = real UDP works on your machine.
"""
import socket
import sys
import time
import struct

ADDR = ("127.0.0.1", 9999)


def server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(ADDR)                          # bind to the fixed address
    print(f"server: bound to {ADDR}, waiting for a client...")
    data, client_addr = sock.recvfrom(1024)  # wait for client's hello
    print(f"server: client connected from {client_addr}")
    n = 0
    while True:
        sock.sendto(struct.pack("!I", n), client_addr)
        print(f"server: sent {n}")
        n += 1
        time.sleep(0.1)


def client():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(b"hello", ADDR)              # register with server
    print("client: said hello, waiting for data...")
    while True:
        data, _ = sock.recvfrom(1024)
        (n,) = struct.unpack("!I", data)
        print(f"client: received {n}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("server", "client"):
        print("usage: python udp_test.py [server|client]")
        sys.exit(1)
    (server if sys.argv[1] == "server" else client)()