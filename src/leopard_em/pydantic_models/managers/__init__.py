"""Pydantic models for Leopard-EM program managers."""

from typing import Any

from .constrained_search_manager import ConstrainedSearchManager
from .frame_inspection_manager import FrameInspectionManager
from .match_template_manager import MatchTemplateManager
from .peak_inspection_manager import PeakInspectionManager
from .refine_template_manager import RefineTemplateManager

__all__ = [
    "ConstrainedSearchManager",
    "FrameInspectionManager",
    "MatchTemplateManager",
    "OptimizeTemplateManager",
    "PeakInspectionManager",
    "RefineTemplateManager",
]


def __getattr__(name: str) -> Any:
    """Import ``OptimizeTemplateManager`` only when it is actually asked for.

    It is the one manager that needs ``ttsim3d``, and ``ttsim3d`` is only needed to
    simulate template volumes. Importing it eagerly means an unusable ``ttsim3d``
    installation also takes down ``match_template``, ``refine_template`` and every
    other manager, none of which touch it.
    """
    if name == "OptimizeTemplateManager":
        from .optimize_template_manager import (
            OptimizeTemplateManager,
        )

        return OptimizeTemplateManager

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
