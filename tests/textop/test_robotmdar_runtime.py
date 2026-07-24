from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import cast

import torch

from robotmdar_textop.runtime import (
    _TEXT_EMBEDDING_CACHE_SIZE,
    RobotMdarGenerator,
    RobotMdarRuntime,
)


class _FakeRobotMdarRuntime:
    torch = torch

    def __init__(self) -> None:
        self.encoded_prompts: list[str] = []
        self.generated_embeddings: list[torch.Tensor] = []

    def encode_text(self, clip_model, prompts: list[str]) -> torch.Tensor:
        del clip_model
        self.encoded_prompts.append(prompts[0])
        return torch.tensor([[len(self.encoded_prompts)]], dtype=torch.float64)

    def generate_next_motion(self, **kwargs):
        self.generated_embeddings.append(kwargs["text_embedding"])
        future_motion = torch.zeros((1, 1, 1), dtype=torch.float32)
        motion_dict = {
            "dof_pos": torch.zeros((1, 1, 23), dtype=torch.float32),
            "dof_vel": torch.zeros((1, 1, 23), dtype=torch.float32),
            "root_rot": torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32),
            "root_trans_offset": torch.zeros((1, 1, 3), dtype=torch.float32),
        }
        return future_motion, motion_dict, SimpleNamespace()


def _generator(runtime: _FakeRobotMdarRuntime) -> RobotMdarGenerator:
    return RobotMdarGenerator(
        runtime=cast(RobotMdarRuntime, runtime),
        clip_model=object(),
        val_data=object(),
        vae=object(),
        cfg_denoiser=object(),
        diffusion=object(),
        history_motion=torch.zeros((1, 1, 1), dtype=torch.float32),
        history_len=1,
        future_len=1,
        abs_pose=object(),
    )


def _next_block(generator: RobotMdarGenerator, prompt: str, index: int) -> None:
    generator.next_block(
        prompt=prompt,
        index=index,
        guidance_scale=5.0,
    )


def test_robotmdar_generator_reuses_exact_prompt_embedding() -> None:
    runtime = _FakeRobotMdarRuntime()
    generator = _generator(runtime)

    _next_block(generator, "stand", 0)
    _next_block(generator, "stand", 1)

    assert runtime.encoded_prompts == ["stand"]
    assert runtime.generated_embeddings[0] is runtime.generated_embeddings[1]
    assert runtime.generated_embeddings[0].dtype == torch.float32


def test_robotmdar_generator_evicts_least_recently_used_embedding() -> None:
    runtime = _FakeRobotMdarRuntime()
    generator = _generator(runtime)
    prompts = [f"prompt {index}" for index in range(_TEXT_EMBEDDING_CACHE_SIZE)]

    for index, prompt in enumerate(prompts):
        _next_block(generator, prompt, index)
    _next_block(generator, prompts[0], len(prompts))
    _next_block(generator, "new prompt", len(prompts) + 1)
    _next_block(generator, prompts[0], len(prompts) + 2)
    _next_block(generator, prompts[1], len(prompts) + 3)

    counts = Counter(runtime.encoded_prompts)
    assert counts[prompts[0]] == 1
    assert counts[prompts[1]] == 2
    assert len(generator._text_embeddings) == _TEXT_EMBEDDING_CACHE_SIZE
