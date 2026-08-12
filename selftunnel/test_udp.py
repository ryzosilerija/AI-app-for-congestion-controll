"""E2E smoke test, UDP mode: friend -> relay -> transport -> client -> echo game."""
import socket, threading, time, subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))

def udp_echo(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", port))
    while True:
        d, a = s.recvfrom(2048)
        s.sendto(b"ECHO:" + d, a)

def main():
    GAME, PUBLIC, CTRL = 7200, 7002, 9020
    threading.Thread(target=udp_echo, args=(GAME,), daemon=True).start()
    time.sleep(0.3)
    relay = subprocess.Popen([sys.executable,"relay.py","--mode","udp",
        "--public-port",str(PUBLIC),"--ctrl-port",str(CTRL)], cwd=HERE)
    time.sleep(1.0)
    client = subprocess.Popen([sys.executable,"client.py","--relay","127.0.0.1",
        "--mode","udp","--local-port",str(GAME),"--ctrl-port",str(CTRL)], cwd=HERE)
    time.sleep(1.5)
    ok=False
    try:
        f = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); f.settimeout(5)
        f.sendto(b"ping-udp", ("127.0.0.1", PUBLIC))
        reply,_ = f.recvfrom(2048)
        print("friend received:", reply)
        ok = reply == b"ECHO:ping-udp"
    finally:
        client.terminate(); relay.terminate()
    print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)

if __name__ == "__main__": main()
