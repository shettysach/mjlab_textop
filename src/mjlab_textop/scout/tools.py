from __future__ import annotations

import json
from base64 import b64encode
from dataclasses import asdict, dataclass
from typing import Protocol

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent, ToolAnnotations

from mjlab_textop.scout.schemas import (
    CapturedView,
    SceneSummary,
    ScoutView,
    TaskInfo,
)

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
STATEFUL = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    openWorldHint=False,
)


class ScoutRuntimeApi(Protocol):
    def list_tasks(self) -> tuple[TaskInfo, ...]: ...

    def load_task(self, task: str) -> TaskInfo: ...

    def get_scene_summary(self) -> SceneSummary: ...

    def capture_view(self, view: ScoutView = "agent") -> CapturedView: ...

    def close_task(self) -> None: ...


@dataclass(frozen=True)
class ScoutTools:
    runtime: ScoutRuntimeApi

    def list_tasks(self) -> tuple[TaskInfo, ...]:
        """List task environments available for inspection."""
        return self.runtime.list_tasks()

    def load_task(self, task: str) -> TaskInfo:
        """Load one task environment, replacing any currently loaded task."""
        return self.runtime.load_task(task)

    def get_scene_summary(self) -> SceneSummary:
        """Describe the objective, robot, and task geometry."""
        return self.runtime.get_scene_summary()

    def capture_view(
        self, view: ScoutView = "agent"
    ) -> list[TextContent | ImageContent]:
        """Capture an agent, overview, or overhead image of the loaded task."""
        captured = self.runtime.capture_view(view)
        metadata = asdict(captured)
        del metadata["image"]
        return [
            TextContent(type="text", text=json.dumps(metadata, separators=(",", ":"))),
            ImageContent(
                type="image",
                data=b64encode(captured.image).decode("ascii"),
                mimeType=captured.mime_type,
            ),
        ]

    def close_task(self) -> str:
        """Close the loaded task and release its renderer."""
        self.runtime.close_task()
        return "Task closed."


def register_tools(mcp: FastMCP, runtime: ScoutRuntimeApi) -> ScoutTools:
    tools = ScoutTools(runtime)
    mcp.tool(annotations=READ_ONLY)(tools.list_tasks)
    mcp.tool(annotations=STATEFUL)(tools.load_task)
    mcp.tool(annotations=READ_ONLY)(tools.get_scene_summary)
    mcp.tool(annotations=READ_ONLY, structured_output=False)(tools.capture_view)
    mcp.tool(annotations=STATEFUL)(tools.close_task)
    return tools
