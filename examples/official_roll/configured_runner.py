"""Optional ROLL YAML-loadable runner using an operator-owned env factory.

The source checkout must be available on the ROLL worker's Python path. No
official ROLL code is vendored here, and importing this module requires ROLL.
"""

from __future__ import annotations

import importlib

from roll.pipeline.agentic.agent_runner.gem_runner import GEMRunner

from .guarded_runner import guarded_gem_runner_class


def resolve_env_factory(path):
    if not isinstance(path, str) or ":" not in path:
        raise ValueError("fpb_env_factory must be an import path module:function")
    module_name, attr_name = path.split(":", 1)
    if (not module_name or not attr_name or not module_name.split(".")
            or not all(part.isidentifier() for part in module_name.split("."))
            or not attr_name.isidentifier()):
        raise ValueError("fpb_env_factory must be an import path module:function")
    factory = getattr(importlib.import_module(module_name), attr_name)
    if not callable(factory):
        raise TypeError("fpb_env_factory must resolve to a callable")
    return factory


_GuardedGEMRunner = guarded_gem_runner_class(GEMRunner)


class VerifiedRealWorldGEMRunner(_GuardedGEMRunner):
    """Resolve a trusted env factory from ROLL's supported config path."""

    def __init__(self, *args, env_config, **kwargs):
        config = env_config.get("config", {})
        factory = resolve_env_factory(config.get("fpb_env_factory"))
        super().__init__(*args, env_config=env_config, env_factory=factory,
                         **kwargs)
