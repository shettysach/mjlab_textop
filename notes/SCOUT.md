```bash
./build/bin/llama-server \
  -m ./models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
  --mmproj ./models/mmproj-F16.gguf \
  --alias gemma-4-E4B-it \
  --host 127.0.0.1 \
  --port 9379 \
  --parallel 1 \
  --n-gpu-layers all \
  --mmproj-offload \
  --flash-attn on \
  --ctx-size 8192 \
  --cache-prompt \
  --cache-reuse 256 \
  --threads 4 \
  --threads-batch 8 \
  --reasoning on \
  --reasoning-budget -1 \
  --n-predict -1 \
  --metrics \
  --perf
```

# Scout: launch and test guide

Scout is a local MCP server for Phase 1. The harness starts it over stdio, the
VLM inspects one task using the Scout skill, and the VLM writes `TASK.md` for a
separate Phase 2 run.

## Install

From the repository root, choose one MJLab device extra and include Scout:

```bash
# NVIDIA GPU
uv sync --extra cu128 --extra scout

# CPU-only alternative
uv sync --extra cpu --extra scout
```

The two device extras conflict, so do not enable both. The MCP SDK version is
already pinned by `uv.lock`.

## Launch command

The server entry point is:

```bash
.venv/bin/mjlab-scout --device cuda:0
```

For CPU testing, use:

```bash
.venv/bin/mjlab-scout --device cpu
```

Optional image-size arguments are `--image-width` and `--image-height`; their
defaults are 640 by 480.

Scout is a stdio server. Running it directly will appear to hang without
printing a prompt because it is waiting for MCP messages on stdin. Normally the
MCP client launches and owns this process; do not start a second copy in another
terminal.

## Connect it to Pi

Install [pi-mcp-adapter](https://pi.dev/packages/pi-mcp-adapter). The project
contains this `.mcp.json` configuration:

```json

  "mcpServers": {
    "mjlab-scout": {
      "command": "uv",
      "args": [
        "run",
        "--extra",
        "cu128",
        "--extra",
        "scout",
        "mjlab-scout"
      ]
    }
  }
}
```

Use `--device cpu` when testing without CUDA. Let the adapter launch the command
rather than launching it yourself. Start Pi from the repository root so the
relative executable path resolves correctly.

The Phase 1 skill is:

```text
.agents/skills/mjlab-scout/SKILL.md
```

Make that skill available to Pi through its skill discovery mechanism. A useful
Phase 1 request is:

```text
Use the mjlab-scout skill to inspect portrait-corridors and write TASK.md in the
current working directory.
```

The expected tool sequence is:

```text
load_task
capture_view("agent")
capture_view for each inspection_* view
capture_view("overview")
close_task
```

`overhead` is available when the route remains ambiguous. For
`portrait-corridors`, `load_task` should advertise `agent`, `overview`,
`overhead`, and three corridor views named `inspection_1` through
`inspection_3`.

## Test the MCP transport without Pi

This smoke test starts Scout as a real stdio child process, loads the corridor
task, requests an MCP image, and saves it to `/tmp/scout-overview.jpg`:

```bash
.venv/bin/python - <<'PY'
import asyncio
import base64
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.types import ImageContent


async def main() -> None:
    root = Path.cwd()
    server = StdioServerParameters(
        command=str(root / ".venv/bin/mjlab-scout"),
        args=["--device", "cuda:0"],
        cwd=root,
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("tools:", [tool.name for tool in tools.tools])

            loaded = await session.call_tool(
                "load_task", {"task": "portrait-corridors"}
            )
            print("task:", loaded.structuredContent)

            captured = await session.call_tool(
                "capture_view", {"view": "overview"}
            )
            image = next(
                block for block in captured.content if isinstance(block, ImageContent)
            )
            output = Path("/tmp/scout-overview.jpg")
            output.write_bytes(base64.b64decode(image.data))
            print("image:", output)

            await session.call_tool("close_task")


asyncio.run(main())
PY
```

Change the smoke-test device to `cpu` if needed. To inspect a corridor portrait
instead, change `overview` to one of the `inspection_*` names returned by
`load_task`.

## Run the automated tests

The focused Scout and camera tests are:

```bash
.venv/bin/pytest -q \
  tests/textop/test_scout.py \
  tests/textop/test_portrait_corridors.py
```

Run the complete project suite with:

```bash
.venv/bin/pytest -q
```

## Verify the Phase 1 output

`TASK.md` should contain exactly these conceptual sections:

```text
# Objective
# Environment
# Success
```

It should describe only visible, qualitative information. It should not contain
coordinates, MJLab implementation names, an oracle route, or Scout tool
instructions. Once it looks correct, start Phase 2 in a clean context and give
that run the generated `TASK.md`.

## Troubleshooting

- No terminal output after launch is normal for a stdio MCP server.
- If Pi cannot see the tools, use an absolute command path and set `cwd` to the
  repository root.
- If `load_task` fails on device initialization, verify the selected `cpu` or
  `cu128` extra and make the `--device` value match it.
- An EGL or OpenGL error means the process cannot access a rendering backend.
  On a headless machine, expose the NVIDIA/EGL devices to the process; for a
  container, this usually means enabling GPU and graphics capabilities.
- MJLab diagnostics go to stderr. Any ordinary text on stdout would corrupt the
  MCP stream.
- Only one task is active per Scout process. `load_task` replaces the previous
  task, while `close_task` releases it explicitly.
