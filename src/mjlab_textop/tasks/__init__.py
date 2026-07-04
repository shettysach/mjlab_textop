from __future__ import annotations

from mjlab_textop.core.task import register_static_task_specs


def register_tasks() -> None:
    from mjlab_textop.tasks.blocked_straight.registration import (
        STATIC_TASK_SPECS as BLOCKED_STRAIGHT_TASK_SPECS,
    )
    from mjlab_textop.tasks.online_textop.registration import (
        STATIC_TASK_SPECS as ONLINE_TEXTOP_TASK_SPECS,
    )
    from mjlab_textop.tasks.straight.registration import (
        STATIC_TASK_SPECS as STRAIGHT_TASK_SPECS,
    )
    from mjlab_textop.tasks.textop_tracking.registration import (
        STATIC_TASK_SPECS as TEXTOP_TRACKING_TASK_SPECS,
    )
    from mjlab_textop.tasks.turn.registration import (
        STATIC_TASK_SPECS as TURN_TASK_SPECS,
    )

    register_static_task_specs(
        [
            *TEXTOP_TRACKING_TASK_SPECS,
            *ONLINE_TEXTOP_TASK_SPECS,
            *STRAIGHT_TASK_SPECS,
            *BLOCKED_STRAIGHT_TASK_SPECS,
            *TURN_TASK_SPECS,
        ]
    )


def ensure_textop_task_registered() -> None:
    register_tasks()
