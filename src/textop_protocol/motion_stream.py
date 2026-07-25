"""Framed binary wire format for live TextOp motion blocks.

The payload keeps motion data as contiguous little-endian ``float32`` values.
Unlike NDJSON, it neither expands every scalar into text nor relies on TCP
packet boundaries or newlines to delimit records.
"""

from __future__ import annotations

import socket
import struct

import numpy as np

from textop_protocol.motion import (
    MotionBlock,
    MotionFrames,
    StreamControl,
    validate_motion_block,
)

_MAGIC = b"TXOP"
_VERSION = 1
_PROMPT_PRESENT = 1
_HEADER = struct.Struct("!4sBBHqqII")
_FLOATS_PER_FRAME = 29 + 29 + 3 + 4
_MAX_FRAMES_PER_BLOCK = 4_096
_MAX_PROMPT_BYTES = 64 * 1024


def textop_block_to_wire(block: MotionBlock) -> bytes:
    """Encode one complete, self-delimiting live motion record."""

    block = validate_motion_block(block)
    prompt = block.control.prompt
    prompt_bytes = b"" if prompt is None else prompt.encode("utf-8")
    if len(prompt_bytes) > _MAX_PROMPT_BYTES:
        raise ValueError(f"Live block prompt exceeds {_MAX_PROMPT_BYTES} bytes")

    frame_count = block.joint_pos.shape[0]
    flags = _PROMPT_PRESENT if prompt is not None else 0
    header = _HEADER.pack(
        _MAGIC,
        _VERSION,
        flags,
        0,
        block.index,
        block.control.recovery_epoch,
        frame_count,
        len(prompt_bytes),
    )
    arrays = (
        block.joint_pos,
        block.joint_vel,
        block.anchor_pos_w,
        block.anchor_quat_w,
    )
    payload = b"".join(
        np.asarray(array, dtype="<f4", order="C").tobytes() for array in arrays
    )
    return header + prompt_bytes + payload


def textop_block_from_wire(record: bytes) -> MotionBlock:
    """Decode one complete live motion record."""

    if len(record) < _HEADER.size:
        raise ValueError("Live block record is shorter than its header")
    magic, version, flags, reserved, index, recovery_epoch, frames, prompt_size = (
        _HEADER.unpack_from(record)
    )
    _validate_header(
        magic=magic,
        version=version,
        flags=flags,
        reserved=reserved,
        frames=frames,
        prompt_size=prompt_size,
    )
    expected_size = _record_size(frames, prompt_size)
    if len(record) != expected_size:
        raise ValueError(
            f"Live block record has {len(record)} bytes; expected {expected_size}"
        )

    prompt_start = _HEADER.size
    prompt_end = prompt_start + prompt_size
    try:
        prompt = record[prompt_start:prompt_end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Live block prompt is not valid UTF-8") from exc
    if not flags & _PROMPT_PRESENT:
        prompt = None

    values = np.frombuffer(record, dtype="<f4", offset=prompt_end)
    joint_end = frames * 29
    velocity_end = joint_end + frames * 29
    position_end = velocity_end + frames * 3
    return validate_motion_block(
        MotionBlock(
            index=index,
            motion=MotionFrames(
                joint_pos=values[:joint_end].reshape(frames, 29),
                joint_vel=values[joint_end:velocity_end].reshape(frames, 29),
                anchor_pos_w=values[velocity_end:position_end].reshape(frames, 3),
                anchor_quat_w=values[position_end:].reshape(frames, 4),
            ),
            control=StreamControl(prompt=prompt, recovery_epoch=recovery_epoch),
        )
    )


def recv_textop_block(sock: socket.socket) -> MotionBlock | None:
    """Read one framed record, or ``None`` after a clean peer disconnect."""

    header = _recv_exact(sock, _HEADER.size, allow_eof=True)
    if header is None:
        return None
    magic, version, flags, reserved, _index, _epoch, frames, prompt_size = (
        _HEADER.unpack(header)
    )
    _validate_header(
        magic=magic,
        version=version,
        flags=flags,
        reserved=reserved,
        frames=frames,
        prompt_size=prompt_size,
    )
    remainder = _recv_exact(
        sock, _record_size(frames, prompt_size) - _HEADER.size, allow_eof=False
    )
    assert remainder is not None
    return textop_block_from_wire(header + remainder)


def _recv_exact(sock: socket.socket, size: int, *, allow_eof: bool) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            if allow_eof and remaining == size:
                return None
            raise ValueError("Live motion stream ended in the middle of a record")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _validate_header(
    *,
    magic: bytes,
    version: int,
    flags: int,
    reserved: int,
    frames: int,
    prompt_size: int,
) -> None:
    if magic != _MAGIC:
        raise ValueError("Unsupported live motion stream magic")
    if version != _VERSION:
        raise ValueError(f"Unsupported live motion stream version: {version}")
    if flags & ~_PROMPT_PRESENT:
        raise ValueError(f"Unsupported live motion stream flags: {flags}")
    if reserved != 0:
        raise ValueError("Live motion stream reserved header bits must be zero")
    if not 0 < frames <= _MAX_FRAMES_PER_BLOCK:
        raise ValueError(f"Invalid live block frame count: {frames}")
    if prompt_size > _MAX_PROMPT_BYTES:
        raise ValueError(f"Live block prompt exceeds {_MAX_PROMPT_BYTES} bytes")
    if not flags & _PROMPT_PRESENT and prompt_size:
        raise ValueError("Live block has prompt bytes without the prompt flag")


def _record_size(frames: int, prompt_size: int) -> int:
    return _HEADER.size + prompt_size + frames * _FLOATS_PER_FRAME * 4
