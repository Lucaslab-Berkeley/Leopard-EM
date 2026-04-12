"""Register minimal ``sys.modules`` entries so imports skip heavy ``__init__.py`` files.

Used by spatial CTF tests when optional Leopard-EM dependencies are missing or
when ``leopard_em.utils`` would otherwise import optional submodules.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


def install_minimal_leopard_em_packages() -> None:
    """Insert ``src`` on path and stub ``leopard_em``/``leopard_em.utils`` packages."""
    s = str(_SRC_ROOT)
    if s not in sys.path:
        sys.path.insert(0, s)

    if "leopard_em" not in sys.modules:
        em = types.ModuleType("leopard_em")
        em.__path__ = [str(_SRC_ROOT / "leopard_em")]
        sys.modules["leopard_em"] = em

    if "leopard_em.utils" not in sys.modules:
        utils_pkg = types.ModuleType("leopard_em.utils")
        utils_pkg.__path__ = [str(_SRC_ROOT / "leopard_em" / "utils")]
        sys.modules["leopard_em.utils"] = utils_pkg


def install_minimal_pydantic_models_packages() -> None:
    """Stub ``leopard_em.pydantic_models`` tree to load individual config modules."""
    install_minimal_leopard_em_packages()
    if "leopard_em.pydantic_models" not in sys.modules:
        pm = types.ModuleType("leopard_em.pydantic_models")
        pm.__path__ = [str(_SRC_ROOT / "leopard_em" / "pydantic_models")]
        sys.modules["leopard_em.pydantic_models"] = pm

    if "leopard_em.pydantic_models.config" not in sys.modules:
        cfg = types.ModuleType("leopard_em.pydantic_models.config")
        cfg.__path__ = [str(_SRC_ROOT / "leopard_em" / "pydantic_models" / "config")]
        sys.modules["leopard_em.pydantic_models.config"] = cfg

    if "leopard_em.pydantic_models.data_structures" not in sys.modules:
        ds = types.ModuleType("leopard_em.pydantic_models.data_structures")
        ds.__path__ = [
            str(_SRC_ROOT / "leopard_em" / "pydantic_models" / "data_structures")
        ]
        sys.modules["leopard_em.pydantic_models.data_structures"] = ds
