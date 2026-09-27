"""Trust-boundary checks for the optional pinned official grader wrapper."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from examples.official_mini_sandbox import strict_grade
from future_prediction_bench.realworld import AdapterInfrastructureError


class FakeOfficialDeployment:
    def __init__(self, *, success=True, status=None, error=None):
        self._config = SimpleNamespace(data_type="swebench")
        self.ds = {
            "instance_id": strict_grade.INSTANCE_ID,
            "repo": strict_grade.REPO,
            "version": strict_grade.VERSION,
            "base_commit": strict_grade.BASE_COMMIT,
            "test_patch": "host-only-test-patch",
        }
        self.test_spec = SimpleNamespace(
            instance_id=strict_grade.INSTANCE_ID,
            repo=strict_grade.REPO,
            version=strict_grade.VERSION,
            FAIL_TO_PASS=["f2p"],
            PASS_TO_PASS=["p2p"],
        )
        self.success = success
        self.status = status or {"f2p": "PASSED", "p2p": "PASSED"}
        self.error = error
        self.legacy_called = False

    def _calculate_reward(self, *args, **kwargs):
        self.legacy_called = True
        raise AssertionError("The masking wrapper must never be used")

    def _calculate_reward_swebench(self, *, get_test_output, timeout):
        assert get_test_output is True
        assert timeout == 300
        if self.error is not None:
            raise self.error
        return self.success, 1.0 if self.success else 0.0, 1.0, (
            f"{strict_grade.START}\nf2p PASSED\np2p PASSED\n{strict_grade.END}\n"
        )

    def get_logs_eval(self, test_spec, output):
        assert test_spec is self.test_spec
        return self.status, True


class MiniGradeTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(
            strict_grade, "_upstream_grade_types",
            return_value=(FakeOfficialDeployment, frozenset()),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_valid_success_requires_all_expected_test_results(self):
        deployment = FakeOfficialDeployment()
        grade = strict_grade.grade_flask_5014(deployment)
        self.assertEqual(grade.reward, 1.0)
        self.assertEqual(grade.expected_cases, grade.passed_cases)
        self.assertEqual(grade.expected_cases, 2)
        self.assertEqual(grade.failed_cases, 0)
        self.assertFalse(deployment.legacy_called)

    def test_real_test_failure_can_be_zero_reward(self):
        deployment = FakeOfficialDeployment(
            success=False, status={"f2p": "FAILED", "p2p": "PASSED"}
        )
        grade = strict_grade.grade_flask_5014(deployment)
        self.assertEqual((grade.reward, grade.failed_cases), (0.0, 1))
        self.assertFalse(deployment.legacy_called)

    def test_upstream_exception_is_ungraded_not_zero_reward(self):
        deployment = FakeOfficialDeployment(error=TimeoutError("test execution hung"))
        with self.assertRaisesRegex(AdapterInfrastructureError, "grader_infrastructure_error"):
            strict_grade.grade_flask_5014(deployment)
        self.assertFalse(deployment.legacy_called)

    def test_missing_expected_case_rejects_upstream_silent_success(self):
        deployment = FakeOfficialDeployment(status={"p2p": "PASSED"})
        with self.assertRaisesRegex(AdapterInfrastructureError, "expected_test_not_observed"):
            strict_grade.grade_flask_5014(deployment)

    def test_skip_list_and_wrong_binding_never_grade(self):
        deployment = FakeOfficialDeployment()
        with mock.patch.object(
            strict_grade, "_upstream_grade_types",
            return_value=(FakeOfficialDeployment, frozenset({strict_grade.INSTANCE_ID})),
        ):
            with self.assertRaisesRegex(AdapterInfrastructureError, "skip_listed"):
                strict_grade.grade_flask_5014(deployment)
        deployment.ds["base_commit"] = "not-the-pinned-base"
        with self.assertRaisesRegex(AdapterInfrastructureError, "task_binding_mismatch"):
            strict_grade.grade_flask_5014(deployment)

    def test_disagreeing_upstream_reward_is_ungraded(self):
        deployment = FakeOfficialDeployment(
            success=True, status={"f2p": "FAILED", "p2p": "PASSED"}
        )
        with self.assertRaisesRegex(AdapterInfrastructureError, "grade_disagreement"):
            strict_grade.grade_flask_5014(deployment)

    def test_disagreeing_upstream_ratios_are_ungraded(self):
        deployment = FakeOfficialDeployment(
            success=False, status={"f2p": "PASSED", "p2p": "FAILED"}
        )
        with self.assertRaisesRegex(AdapterInfrastructureError, "ratio_disagreement"):
            strict_grade.grade_flask_5014(deployment)


if __name__ == "__main__":
    unittest.main()
