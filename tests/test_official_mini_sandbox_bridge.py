import asyncio
import sys
import types
import unittest
from unittest.mock import patch

from examples.official_mini_sandbox.session_bridge import (
    MiniSandboxUnavailable, run_in_official_session,
)


class TestOfficialMiniSandboxBridge(unittest.TestCase):
    def _upstream_fake(self):
        recorded = []

        class BashAction:
            def __init__(self, *, command, timeout, check):
                recorded.append((command, timeout, check))

        class Session:
            async def run_in_session(self, action):
                self.action = action
                return types.SimpleNamespace(output="visible-check-ok\n")

        class SandboxDeployment:
            def __init__(self):
                self.runtime = Session()

        modules = {
            "swesandbox": types.ModuleType("swesandbox"),
            "swesandbox.sandbox_deployment": types.ModuleType("swesandbox.sandbox_deployment"),
            "swerex": types.ModuleType("swerex"),
            "swerex.deployment": types.ModuleType("swerex.deployment"),
            "swerex.deployment.config": types.ModuleType("swerex.deployment.config"),
            "swerex.runtime": types.ModuleType("swerex.runtime"),
            "swerex.runtime.abstract": types.ModuleType("swerex.runtime.abstract"),
        }
        modules["swesandbox.sandbox_deployment"].SandboxDeployment = SandboxDeployment
        modules["swerex.runtime.abstract"].BashAction = BashAction
        return modules, SandboxDeployment, recorded

    def test_invokes_verified_upstream_action_contract(self):
        modules, Deployment, recorded = self._upstream_fake()
        with patch.object(sys, "platform", "linux"), patch.dict(sys.modules, modules):
            deployment = Deployment()
            output = asyncio.run(run_in_official_session(
                deployment, operator_command="python -B -m pytest -q", timeout_seconds=12))
        self.assertEqual(output, "visible-check-ok\n")
        self.assertEqual(recorded, [("python -B -m pytest -q", 12, "raise")])
        self.assertIsInstance(deployment.runtime.action, modules["swerex.runtime.abstract"].BashAction)

    def test_rejects_non_upstream_deployment(self):
        modules, _, _ = self._upstream_fake()
        with patch.object(sys, "platform", "linux"), patch.dict(sys.modules, modules):
            with self.assertRaises(TypeError):
                asyncio.run(run_in_official_session(object(), operator_command="true"))

    def test_non_linux_fails_before_import_or_execution(self):
        with patch.object(sys, "platform", "darwin"):
            with self.assertRaises(MiniSandboxUnavailable):
                asyncio.run(run_in_official_session(object(), operator_command="true"))

    def test_missing_dependency_fails_closed(self):
        with patch.object(sys, "platform", "linux"), patch.dict(sys.modules,
                {"swesandbox": None, "swesandbox.sandbox_deployment": None}):
            with self.assertRaises(MiniSandboxUnavailable):
                asyncio.run(run_in_official_session(object(), operator_command="true"))


if __name__ == "__main__":
    unittest.main()
