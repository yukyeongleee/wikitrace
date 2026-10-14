// In-container half of the traverse `allowlist` egress control.
//
// The container runs at `--network none`, so Pi's `fetch` to openrouter.ai:443
// has no route. This listener restores exactly one path: an `/etc/hosts` pin
// (--add-host) resolves each allowlisted host to a loopback IP we listen on, and
// we pipe those bytes verbatim across a bind-mounted unix socket to the host-side
// relay (see egress.py), which forwards to the real host. Pure byte relay — the
// bytes are TLS ciphertext, never decrypted here.
//
// Usage: node incontainer_relay.mjs <config.json> -- <pi-cmd> [pi-args...]
//   config.json: [{ "ip": "127.0.0.1", "port": 443, "sock": "/net/openrouter.ai.sock" }, ...]
// We start every listener, THEN spawn Pi with stdio inherited (Pi speaks RPC on
// stdin/stdout, so this process must never write there — all logs go to stderr).
// We exit with Pi's exit code; the listeners die with the process.

import net from "node:net";
import { readFileSync } from "node:fs";
import { spawn } from "node:child_process";

const argv = process.argv.slice(2);
const sep = argv.indexOf("--");
if (sep === -1) {
  console.error("[egress-relay] missing '--' separating config from the Pi command");
  process.exit(2);
}
const configPath = argv[0];
const piCmd = argv.slice(sep + 1);
if (!configPath || piCmd.length === 0) {
  console.error("[egress-relay] usage: relay <config.json> -- <pi-cmd> [args...]");
  process.exit(2);
}

let entries;
try {
  entries = JSON.parse(readFileSync(configPath, "utf8"));
} catch (err) {
  console.error(`[egress-relay] cannot read config ${configPath}: ${err.message}`);
  process.exit(2);
}

// Bind every listener before Pi starts, so its first LLM call never races setup.
// A bind failure is fatal (fail loud) — a missing hole must not look like a Pi
// failure downstream.
let pending = entries.length;
function listenerReady() {
  if (--pending === 0) startPi();
}

for (const { ip, port, sock } of entries) {
  const server = net.createServer((downstream) => {
    const upstream = net.connect(sock);
    // Either side closing tears the pair down; errors are expected on close.
    const kill = () => {
      downstream.destroy();
      upstream.destroy();
    };
    downstream.on("error", kill);
    upstream.on("error", kill);
    downstream.pipe(upstream);
    upstream.pipe(downstream);
  });
  server.on("error", (err) => {
    console.error(`[egress-relay] listen ${ip}:${port} failed: ${err.message}`);
    process.exit(2);
  });
  server.listen(port, ip, listenerReady);
}

if (entries.length === 0) startPi(); // no allowlist -> nothing to relay; run Pi

function startPi() {
  const child = spawn(piCmd[0], piCmd.slice(1), { stdio: "inherit" });
  child.on("exit", (code, signal) => {
    process.exit(signal ? 1 : code ?? 0);
  });
  child.on("error", (err) => {
    console.error(`[egress-relay] failed to spawn Pi: ${err.message}`);
    process.exit(1);
  });
}
