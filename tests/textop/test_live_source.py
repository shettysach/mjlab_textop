from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np
import pytest
from builders import motion_block

from mjlab_textop.core.online import block, source, wire
from mjlab_textop.core.online.live import (
    SocketOnlineSource,
    SocketSourceCfg,
    parse_textop_block_message,
    textop_block_to_ndjson_message,
)
from mjlab_textop.core.online.source import StreamControl


def test_legacy_online_imports_reexport_canonical_objects() -> None:
    assert source.MotionBlock is block.MotionBlock
    assert source.MotionFrames is block.MotionFrames
    assert source.StreamControl is block.StreamControl
    assert source.validate_motion_block is block.validate_motion_block
    assert parse_textop_block_message is wire.parse_textop_block_message
    assert textop_block_to_ndjson_message is wire.textop_block_to_ndjson_message


def test_textop_block_ndjson_round_trip() -> None:
    block = replace(
        motion_block(index=100, frames=8),
        control=StreamControl(prompt="stand", recovery_epoch=3),
    )

    message = textop_block_to_ndjson_message(block)
    parsed = parse_textop_block_message(message)

    assert '"fps"' not in message
    assert parsed.index == 100
    assert parsed.control.prompt == "stand"
    assert parsed.control.recovery_epoch == 3
    np.testing.assert_allclose(parsed.joint_pos, block.joint_pos)
    np.testing.assert_allclose(parsed.joint_vel, block.joint_vel)
    np.testing.assert_allclose(parsed.anchor_pos_w, block.anchor_pos_w)
    np.testing.assert_allclose(
        parsed.anchor_quat_w,
        np.tile([1.0, 0.0, 0.0, 0.0], (8, 1)),
    )


def test_textop_block_parser_rejects_missing_field() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        parse_textop_block_message({"index": 0})


def test_textop_block_parser_rejects_bad_shape() -> None:
    block = motion_block(index=0, frames=8)
    message = {
        "index": 0,
        "joint_pos": np.zeros((8, 28), dtype=np.float32).tolist(),
        "joint_vel": block.joint_vel.tolist(),
        "anchor_pos_w": block.anchor_pos_w.tolist(),
        "anchor_quat_w": block.anchor_quat_w.tolist(),
    }

    with pytest.raises(ValueError, match="29 joints"):
        parse_textop_block_message(message)


def test_socket_source_blocks_when_queue_is_full() -> None:
    source = SocketOnlineSource(SocketSourceCfg(max_queue_blocks=1))
    source.append_message(textop_block_to_ndjson_message(motion_block(index=0)))

    started = threading.Event()

    def append_second_block() -> None:
        started.set()
        source.append_message(textop_block_to_ndjson_message(motion_block(index=8)))

    thread = threading.Thread(target=append_second_block)
    thread.start()
    assert started.wait(timeout=1.0)
    thread.join(timeout=0.05)
    assert thread.is_alive()

    assert source.diagnostics.queue_depth == 1

    block = source.poll()

    assert block is not None
    assert block.index == 0
    thread.join(timeout=1.0)
    assert not thread.is_alive()

    block = source.poll()

    assert block is not None
    assert block.index == 8
    assert source.diagnostics.blocks_received == 2
    assert source.diagnostics.blocks_polled == 2
    assert source.diagnostics.queue_depth == 0


def test_socket_source_records_bad_messages() -> None:
    source = SocketOnlineSource()

    source._handle_line(b"{not json}\n")

    assert source.diagnostics.bad_messages == 1
    assert source.diagnostics.last_error is not None
