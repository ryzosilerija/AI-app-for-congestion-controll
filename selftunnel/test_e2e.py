"""End-to-end smoke test on localhost, TCP mode.

Topology (all local, but exercising the real transport + relay + client):

  friend  --tcp-->  relay :7001  ==transport==>  client  --tcp-->  echo game :7100

We start an echo 'game', the relay, the client, then connect as the 'friend'
and check the echo round-trips through the whole tunnel.
"""
import socket, threading, time, subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))


def echo_game(port):
    ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("127.0.0.1", port)); ls.listen(4)
    while True:
        c, _ = ls.accept()
        def handle(c=c):
            while True:
                d = c.recv(4096)
                if not d: break
                c.sendall(b"ECHO:" + d)
        threading.Thread(target=handle, daemon=True).start()


def main():
    GAME, PUBLIC, CTRL = 7100, 7001, 9010
    threading.Thread(target=echo_game, args=(GAME,), daemon=True).start()
    time.sleep(0.3)

    relay = subprocess.Popen(
        [sys.executable, "relay.py", "--mode", "tcp",
         "--public-port", str(PUBLIC), "--ctrl-port", str(CTRL)],
        cwd=HERE)
    time.sleep(1.0)
    client = subprocess.Popen(
        [sys.executable, "client.py", "--relay", "127.0.0.1", "--mode", "tcp",
         "--local-port", str(GAME), "--ctrl-port", str(CTRL)],
        cwd=HERE)
    time.sleep(1.5)

    ok = False
    try:
        f = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        f.settimeout(5)
        f.connect(("127.0.0.1", PUBLIC))
        f.sendall(b"hello-tunnel")
        reply = f.recv(4096)
        print("friend received:", reply)
        ok = reply == b"ECHO:hello-tunnel"
        f.close()
    finally:
        client.terminate(); relay.terminate()

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
