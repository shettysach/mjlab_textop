from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np
import pytest
from builders import motion_block

from mjlab_textop.core.online.live import (
    SocketOnlineSource,
    SocketSourceCfg,
    textop_block_from_wire,
    textop_block_to_wire,
)
from mjlab_textop.core.online.source import StreamControl


def test_textop_block_binary_round_trip() -> None:
    block = replace(
        motion_block(index=100, frames=8),
        control=StreamControl(prompt="stand", recovery_epoch=3, checkpoint_id=9),
    )

    record = textop_block_to_wire(block)
    parsed = textop_block_from_wire(record)

    assert record[:4] == b"TXOP"
    assert parsed.index == 100
    assert parsed.control.prompt == "stand"
    assert parsed.control.recovery_epoch == 3
    assert parsed.control.checkpoint_id == 9
    assert parsed.joint_pos.flags.writeable
    assert parsed.joint_vel.flags.writeable
    assert parsed.anchor_pos_w.flags.writeable
    assert parsed.anchor_quat_w.flags.writeable
    np.testing.assert_allclose(parsed.joint_pos, block.joint_pos)
    np.testing.assert_allclose(parsed.joint_vel, block.joint_vel)
    np.testing.assert_allclose(parsed.anchor_pos_w, block.anchor_pos_w)
    np.testing.assert_allclose(
        parsed.anchor_quat_w,
        np.tile([1.0, 0.0, 0.0, 0.0], (8, 1)),
    )


def test_textop_block_parser_rejects_truncated_record() -> None:
    with pytest.raises(ValueError, match="shorter than its header"):
        textop_block_from_wire(b"TXOP")


def test_textop_block_parser_rejects_corrupt_magic() -> None:
    record = bytearray(textop_block_to_wire(motion_block(index=0, frames=8)))
    record[:4] = b"NOPE"

    with pytest.raises(ValueError, match="magic"):
        textop_block_from_wire(bytes(record))


def test_textop_block_parser_rejects_old_protocol_version() -> None:
    record = bytearray(textop_block_to_wire(motion_block(index=0, frames=8)))
    record[4] = 1

    with pytest.raises(ValueError, match="Unsupported live motion stream version: 1"):
        textop_block_from_wire(bytes(record))


def test_socket_source_blocks_when_queue_is_full() -> None:
    source = SocketOnlineSource(SocketSourceCfg(max_queue_blocks=1))
    source.append_wire_record(textop_block_to_wire(motion_block(index=0)))

    started = threading.Event()

    def append_second_block() -> None:
        started.set()
        source.append_wire_record(textop_block_to_wire(motion_block(index=8)))

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


def test_socket_source_records_bad_records() -> None:
    source = SocketOnlineSource()

    class BrokenSocket:
        called = False

        def recv(self, _size: int) -> bytes:
            if self.called:
                return b""
            self.called = True
            return b"NOPE"

    source._handle_next_record(BrokenSocket())

    assert source.diagnostics.bad_messages == 1
    assert source.diagnostics.last_error is not None
