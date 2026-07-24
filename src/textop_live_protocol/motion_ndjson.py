from __future__ import annotations

import json
from typing import Any

import numpy as np

from textop_live_protocol.motion import (
    MotionBlock,
    MotionFrames,
    StreamControl,
    validate_motion_block,
)


def textop_block_to_ndjson_message(block: MotionBlock) -> str:
    message = {
        "index": int(block.index),
        "joint_pos": np.asarray(block.joint_pos, dtype=np.float32).tolist(),
        "joint_vel": np.asarray(block.joint_vel, dtype=np.float32).tolist(),
        "anchor_pos_w": np.asarray(block.anchor_pos_w, dtype=np.float32).tolist(),
        "anchor_quat_w": np.asarray(block.anchor_quat_w, dtype=np.float32).tolist(),
        "recovery_epoch": int(block.control.recovery_epoch),
    }
    if block.control.prompt is not None:
        message["prompt"] = block.control.prompt
    return json.dumps(message, separators=(",", ":")) + "\n"


def parse_textop_block_message(message: str | bytes | dict[str, Any]) -> MotionBlock:
    data = _load_message(message)
    missing = [
        key
        for key in (
            "index",
            "joint_pos",
            "joint_vel",
            "anchor_pos_w",
            "anchor_quat_w",
        )
        if key not in data
    ]
    if missing:
        raise ValueError(f"Live block missing required fields: {missing}")

    return validate_motion_block(
        MotionBlock(
            index=int(data["index"]),
            motion=MotionFrames(
                joint_pos=np.asarray(data["joint_pos"]),
                joint_vel=np.asarray(data["joint_vel"]),
                anchor_pos_w=np.asarray(data["anchor_pos_w"]),
                anchor_quat_w=np.asarray(data["anchor_quat_w"]),
            ),
            control=StreamControl(
                prompt=None if data.get("prompt") is None else str(data["prompt"]),
                recovery_epoch=data.get("recovery_epoch", 0),
            ),
        )
    )


def _load_message(message: str | bytes | dict[str, Any]) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    data = json.loads(message)
    if not isinstance(data, dict):
        raise ValueError("TextOp live block message must be a JSON object")
    return data
