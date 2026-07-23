from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP

from mjlab_textop.scout.config import ScoutConfig
from mjlab_textop.scout.runtime import ScoutRuntime
from mjlab_textop.scout.tools import register_tools


def create_server(runtime: ScoutRuntime) -> FastMCP:
    server = FastMCP("mjlab-scout")
    register_tools(server, runtime)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MJLab Scout MCP server")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    args = parser.parse_args()

    runtime = ScoutRuntime(
        ScoutConfig(
            device=args.device,
            image_width=args.image_width,
            image_height=args.image_height,
        )
    )
    server = create_server(runtime)
    try:
        server.run(transport="stdio")
    finally:
        runtime.close()
