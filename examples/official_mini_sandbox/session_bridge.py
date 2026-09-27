"""Invoke an existing official SWE-MiniSandbox / SWE-ReX terminal session.

This does not create a sandbox or implement our RealWorldEnv adapter. The
caller must initialize and attach the official SandboxDeployment through its
supported Linux SWE-Agent/SWE-ReX flow before calling this bridge. It is an
optional direct API call, with no vendored upstream source.
"""

from __future__ import annotations

import importlib
import sys


class MiniSandboxUnavailable(RuntimeError):
    """The pinned upstream Linux integration is not installed or available."""


def _official_types():
    if sys.platform != "linux":
        raise MiniSandboxUnavailable("Official SWE-MiniSandbox requires Linux namespaces")
    try:
        # At the pinned upstream revision, SWE-ReX config registers
        # SandboxDeploymentConfig. Importing the sandbox module first enters
        # that registration while config is only partially initialized.
        importlib.import_module("swerex.deployment.config")
        deployment_module = importlib.import_module("swesandbox.sandbox_deployment")
        runtime_module = importlib.import_module("swerex.runtime.abstract")
        return deployment_module.SandboxDeployment, runtime_module.BashAction
    except (ImportError, AttributeError) as exc:
        raise MiniSandboxUnavailable(
            "Install a pinned official SWE-MiniSandbox checkout and its SWE-ReX dependencies"
        ) from exc


async def run_in_official_session(deployment, *, operator_command: str,
                                  timeout_seconds: int = 30, max_output_chars: int = 16000) -> str:
    """Run one operator-authored command through upstream ``run_in_session``.

    ``deployment`` must be an already attached official SandboxDeployment. Do
    not pass a policy-authored shell command or hidden verifier secrets. A
    live runtime call does not, by itself, prove sandbox isolation or grading.
    """
    Deployment, BashAction = _official_types()
    if not isinstance(deployment, Deployment):
        raise TypeError("deployment must be an official SandboxDeployment")
    if (not isinstance(operator_command, str) or not operator_command.strip()
            or len(operator_command) > 4096 or "\x00" in operator_command):
        raise ValueError("operator_command must be a bounded nonempty string")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 300:
        raise ValueError("timeout_seconds must be an integer in [1, 300]")
    if type(max_output_chars) is not int or not 1 <= max_output_chars <= 100000:
        raise ValueError("max_output_chars must be an integer in [1, 100000]")
    action = BashAction(command=operator_command, timeout=timeout_seconds, check="raise")
    result = await deployment.runtime.run_in_session(action)
    output = result.output
    if not isinstance(output, str) or len(output) > max_output_chars:
        raise ValueError("Upstream session returned invalid or oversized output")
    return output
