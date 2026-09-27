"""ROLL YAML-loadable manager with a verified-reward sample boundary."""

import ray
from roll.pipeline.agentic.env_manager.proxy_env_manager import ProxyEnvManager

from .manager_guard import guarded_proxy_env_manager_class


VerifiedProxyEnvManager = guarded_proxy_env_manager_class(
    ProxyEnvManager, ray_get=ray.get,
)
