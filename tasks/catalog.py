from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

from mjlab.envs import ManagerBasedRlEnvCfg

from tasks.blocked_straight.env_cfg import make_blocked_straight_g1_env_cfg
from tasks.online_textop.env_cfg import make_online_textop_g1_env_cfg
from tasks.portrait_corridors.env_cfg import make_portrait_corridors_g1_env_cfg
from tasks.side_goals.env_cfg import make_side_goals_g1_env_cfg
from tasks.straight.env_cfg import make_straight_g1_env_cfg
from tasks.turn.env_cfg import make_turn_task_g1_env_cfg

TaskSet = Literal[
    "straight",
    "blocked-straight",
    "side-goals",
    "turn",
    "portrait-corridors",
]
EnvCfgFactory = Callable[..., ManagerBasedRlEnvCfg]


@dataclass(frozen=True)
class TaskDefinition:
    env_factory: EnvCfgFactory
    objective: str


TASKS: dict[TaskSet, TaskDefinition] = {
    "straight": TaskDefinition(
        env_factory=make_straight_g1_env_cfg,
        objective="Reach and stand on the green region.",
    ),
    "blocked-straight": TaskDefinition(
        env_factory=make_blocked_straight_g1_env_cfg,
        objective="Reach and stand on the green region.",
    ),
    "side-goals": TaskDefinition(
        env_factory=make_side_goals_g1_env_cfg,
        objective="Reach and stand on the green region.",
    ),
    "turn": TaskDefinition(
        env_factory=make_turn_task_g1_env_cfg,
        objective="Reach and stand on the green region.",
    ),
    "portrait-corridors": TaskDefinition(
        env_factory=make_portrait_corridors_g1_env_cfg,
        objective="Stand in front of the creator of Linux.",
    ),
}


def make_task_env_cfg(task: TaskSet | None, **kwargs: Any) -> ManagerBasedRlEnvCfg:
    factory = make_online_textop_g1_env_cfg if task is None else TASKS[task].env_factory
    return factory(**kwargs)


def get_task(task: str) -> tuple[TaskSet, TaskDefinition]:
    if task not in TASKS:
        available = ", ".join(TASKS)
        raise ValueError(
            f"Unknown Scout task {task!r}. Available: {available}"
        ) from None
    name = cast(TaskSet, task)
    return name, TASKS[name]
