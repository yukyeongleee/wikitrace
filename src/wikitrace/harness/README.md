# Harness

Every method runs the same agent: the Pi coding agent in a Docker sandbox, with the `read` and `bash` tools and one environment tool (`act` on ALFWorld and ScienceWorld, `take_action` on PlanCraft and τ³-bench, `exec` on AppWorld). Methods differ only in the memory placed in the prompt and in what is mounted at `/work/corpus`.

## Pi

`scripts/setup/pi.sh` installs Pi under `tools/pi-mono`. It checks out the fork `jdf-prog/pi-mono` at a pinned commit, applies the patches in `tools/pi-patches`, and builds it. It stops if the commit differs or a patch does not apply. Pi needs Node 20.

| Patch | Change |
|---|---|
| `0001` | records the cost OpenRouter reports for each request |
| `0002` | adds a `get_usage` command so the runner logs per-request usage |
| `0003` | stops counting reasoning tokens twice |
| `0004` | caps each `bash` call at `traverse.cmd_timeout` seconds; the agent sees the timeout and continues |

## Sandbox

`scripts/setup/sandbox_image.sh` builds `wikitrace-sandbox:node20` (Node 20, Python 3, NumPy). Each episode runs in its own container started with `--network none`. The container sees the mounted memory, the two read-only helper commands, the environment tool and an empty scratch directory. Its only way out is a relay on the host that forwards Pi's requests to `openrouter.ai`.

## Environments

Each episode starts an environment server for its benchmark. The server counts the agent's actions, enforces the step budget and reports the outcome.

| Benchmark | Installer | Runs in |
|---|---|---|
| ALFWorld | `scripts/setup/alfworld.sh` | `data/alfworld/.venv` |
| ScienceWorld | `scripts/setup/scienceworld.sh` | `data/scienceworld/.venv` (with a JDK) |
| PlanCraft | `scripts/setup/plancraft.sh` | `data/plancraft/.venv` |
| τ³-bench Retail | `scripts/setup/tau3.sh` | `data/tau3/.venv` |
| AppWorld | `scripts/setup/appworld.sh` | a Docker image built by `scripts/setup/appworld_env_image.sh` |

- **τ³-bench**: the server also runs the simulated user (`openrouter/openai/gpt-4.1`, set in `configs/tau3.yaml`), which calls OpenRouter with `OPENROUTER_API_KEY`.
- **AppWorld**: each episode runs in a fresh container of the episode image, which holds no gold answers. The host grades the episode afterwards, using lenient TGC as in the paper. The run summary also reports the official TGC.
