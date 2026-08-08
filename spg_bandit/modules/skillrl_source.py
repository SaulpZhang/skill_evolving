"""Load the vendored SkillRL modules without depending on ``docs/SkillRL``."""

from __future__ import annotations

import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILLRL_ROOT = PROJECT_ROOT / "resource" / "skillrl"
if not SKILLRL_ROOT.is_dir():
    raise ImportError(
        "SkillRL runtime resources are missing: "
        f"{SKILLRL_ROOT}. Copy the required files into resource/skillrl."
    )


def _register_source_package(name: str, path: Path) -> None:
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    sys.modules.setdefault(name, package)


SOURCE_PACKAGE = "_spg_skillrl_source"
_register_source_package(SOURCE_PACKAGE, SKILLRL_ROOT / "agent_system")
_register_source_package(
    f"{SOURCE_PACKAGE}.memory", SKILLRL_ROOT / "agent_system" / "memory",
)
_register_source_package(
    f"{SOURCE_PACKAGE}.environments",
    SKILLRL_ROOT / "agent_system" / "environments",
)
_register_source_package(
    f"{SOURCE_PACKAGE}.environments.prompts",
    SKILLRL_ROOT / "agent_system" / "environments" / "prompts",
)
_register_source_package(
    f"{SOURCE_PACKAGE}.environments.env_package",
    SKILLRL_ROOT / "agent_system" / "environments" / "env_package",
)
_register_source_package(
    f"{SOURCE_PACKAGE}.environments.env_package.alfworld",
    SKILLRL_ROOT / "agent_system" / "environments" / "env_package" / "alfworld",
)

from _spg_skillrl_source.environments.env_package.alfworld.projection import (  # noqa: E402
    alfworld_projection,
)
from _spg_skillrl_source.environments.prompts.alfworld import (  # noqa: E402
    ALFWORLD_TEMPLATE_NO_HIS,
    ALFWORLD_TEMPLATE_WITH_MEMORY,
)
from _spg_skillrl_source.memory.skill_updater import SkillUpdater  # noqa: E402
from _spg_skillrl_source.memory.skills_only_memory import SkillsOnlyMemory  # noqa: E402


__all__ = [
    "ALFWORLD_TEMPLATE_NO_HIS",
    "ALFWORLD_TEMPLATE_WITH_MEMORY",
    "PROJECT_ROOT",
    "SKILLRL_ROOT",
    "SkillUpdater",
    "SkillsOnlyMemory",
    "alfworld_projection",
]
