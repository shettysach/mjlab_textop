"""MJLab TextOp utilities."""

import os
import sys

# Match the rendering backend to MJLab's viewer before importing MJLab.
if sys.platform.startswith("linux"):
    has_display = (
        os.environ.get("DISPLAY") is not None
        or os.environ.get("WAYLAND_DISPLAY") is not None
    )
    os.environ.setdefault("MUJOCO_GL", "glfw" if has_display else "egl")
