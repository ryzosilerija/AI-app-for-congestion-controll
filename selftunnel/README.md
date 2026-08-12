# selftunnel

A self-hosted tunnel that exposes a local game (or any service) to the internet
over **your own UDP transport** — so **your own congestion control** is in the
path, not the kernel's TCP stack. Supports **TCP or UDP** per tunnel and prints
a shareable `public-ip:port` link.

This is the vehicle for playing over a custom transport: existing tools (ngrok,
Tailscale, plain port-forwarding) all run over kernel TCP/UDP and bypass your
congestion control. This one carries traffic through a userspace transport with
a **pluggable controller**, which is where your PyQUIC-RL agent plugs in.

## How it works

```
friend  --tcp/udp-->  relay (PUBLIC IP)  ==your UDP transport==>  client (your PC)  --tcp/udp-->  game
                       prints the link         congestion control here
```

- **relay** runs on a machine with a public IP. It opens the public port your
  friend connects to, and prints `public-ip:port` to share.
- **client** runs on your machine next to the game. It dials *out* to the relay
  (defeating NAT), and forwards traffic to the local game port.
- Between them, bytes ride your reliable UDP transport, paced by the selected
  congestion controller.

## The link

The link is simply the **relay's public IP and the public port** you chose:
`203.0.113.7:25565`. There's no domain or subdomain — that only applies to
HTTP tunnels. A game link is `ip:port`, which is what the relay prints on start.

**Requirement:** the relay must run somewhere with a public IP (a cheap VPS, or
your machine if it truly has one — many home connections are behind CGNAT and
can't accept inbound connections, which is exactly why the relay exists).

## Run it

On the public box (the relay):
```bash
python relay.py --mode tcp --public-port 25565 --cc aimd
# prints:  SHARE THIS LINK:  <public-ip>:25565  (TCP)
```

On your machine (next to the game on local port 25565):
```bash
python client.py --relay <relay-public-ip> --mode tcp --local-port 25565 --cc aimd
```

Your friend points the game at `<relay-public-ip>:25565`. Swap `--mode udp`
for UDP-native games.

## Plugging in your congestion control

`congestion.py` defines the `CongestionController` interface (`can_send`,
`on_ack`, `on_loss`, `window_bytes`) and a working `AIMD` default. To use your
PyQUIC-RL agent:

1. Wrap it in a class implementing that interface.
2. Register it in `CONTROLLERS` (e.g. `"rl": YourRLController`).
3. Run with `--cc rl`.

The transport calls `on_ack`/`on_loss` as feedback arrives and gates sending on
`can_send`, so your agent drives real pacing over a real link.

## Tests

```bash
python test_e2e.py     # TCP path: friend -> relay -> transport -> client -> echo
python test_udp.py     # UDP path
```

Both spin up the whole chain on localhost and assert an echo round-trips.

## Status / next steps

This is a working v1 reference — single tunnel, no encryption, simple loss
detection. Natural extensions: TLS/encryption, multiple concurrent tunnels,
auth tokens, a control UI, and swapping the reference transport for your full
PyQUIC-RL stack. The seam is built for exactly that swap.
