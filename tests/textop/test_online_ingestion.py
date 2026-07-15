from __future__ import annotations

from dataclasses import replace

import pytest
from builders import motion_block

from mjlab_textop.core.online.buffer import RollingMotionBuffer
from mjlab_textop.core.online.ingestion import OnlineBlockIngestor
from mjlab_textop.core.online.source import QueueOnlineSource, StreamControl


def test_recovery_realigns_producer_indices_and_keeps_mapping_afterward() -> None:
    recovery_block = replace(
        motion_block(index=24, frames=8),
        control=StreamControl(prompt="stand", recovery_epoch=2),
    )
    next_block = motion_block(index=32, frames=8)
    source = QueueOnlineSource([recovery_block])
    buffer = RollingMotionBuffer()
    ingestor = OnlineBlockIngestor(source, buffer, live=True)

    ingestor.begin_recovery()
    ready = ingestor.poll_recovery(
        max_blocks=4,
        current_frame=0,
        future_steps=5,
        accepts=lambda block: block.control.recovery_epoch == 2,
    )

    assert ready is True
    assert ingestor.index_offset == -24
    assert buffer.earliest_index == 0
    assert buffer.latest_index == 7

    source.append(next_block)
    assert ingestor.poll(max_blocks=4) == 1
    assert buffer.latest_index == 15


def test_recovery_discards_blocks_rejected_by_its_policy() -> None:
    rejected = motion_block(index=8, frames=8)
    accepted = replace(
        motion_block(index=16, frames=8),
        control=StreamControl(prompt="stand", recovery_epoch=3),
    )
    buffer = RollingMotionBuffer()
    ingestor = OnlineBlockIngestor(
        QueueOnlineSource([rejected, accepted]),
        buffer,
        live=True,
    )

    ingestor.begin_recovery()
    ready = ingestor.poll_recovery(
        max_blocks=4,
        current_frame=4,
        future_steps=5,
        accepts=lambda block: block.control.recovery_epoch == 3,
    )

    assert ready is True
    assert ingestor.index_offset == -12
    assert buffer.earliest_index == 4
    assert buffer.latest_index == 11


def test_live_ingestion_rejects_noncontiguous_mapped_blocks() -> None:
    source = QueueOnlineSource(
        [motion_block(index=0, frames=8), motion_block(index=9, frames=8)]
    )
    ingestor = OnlineBlockIngestor(source, RollingMotionBuffer(), live=True)

    with pytest.raises(
        RuntimeError,
        match="expected block index 8, got 9",
    ):
        ingestor.poll(max_blocks=2)


def test_live_polling_policy_can_pause_before_consuming_a_block() -> None:
    source = QueueOnlineSource([motion_block(frames=8)])
    buffer = RollingMotionBuffer()
    ingestor = OnlineBlockIngestor(source, buffer, live=True)

    assert ingestor.poll(max_blocks=4, should_poll=lambda: False) == 0
    assert buffer.frame_count == 0
    assert ingestor.poll(max_blocks=4, should_poll=lambda: True) == 1
