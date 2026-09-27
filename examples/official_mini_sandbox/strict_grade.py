"""Fail-closed host-only wrapper for pinned MiniSandbox SWE-bench grading.

The upstream public `_calculate_reward` turns any grader exception into
`(0.0, {}, {}, '')`. A verifier failure must instead remain ungraded. This
module calls the upstream strict method directly and independently checks that
every expected Flask test was actually observed in the parsed log. It must
never be invoked from an actor-accessible session or expose the task dataset
or returned grade to the actor.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import math
import sys

from future_prediction_bench.realworld import AdapterInfrastructureError

from .task_asset_preflight import BASE_COMMIT, INSTANCE_ID, REPO, VERSION


START = ">>>>> Start Test Output"
END = ">>>>> End Test Output"
PASS_STATUSES = frozenset({"PASSED", "XFAIL"})
FAIL_STATUSES = frozenset({"FAILED", "ERROR"})


@dataclass(frozen=True)
class TrustedMiniGrade:
    """Trusted-host evidence; do not append this object to actor observations."""

    reward: float
    instance_id: str
    expected_cases: int
    passed_cases: int
    failed_cases: int
    log_sha256: str


def _upstream_grade_types():
    if sys.platform != "linux":
        raise AdapterInfrastructureError("official_mini_requires_linux")
    try:
        # Registration order matters at the pinned upstream revision.
        importlib.import_module("swerex.deployment.config")
        deployment = importlib.import_module("swesandbox.sandbox_deployment")
        instance_map = importlib.import_module("swesandbox.swe_bench_instance_map")
        return deployment.SandboxDeployment, frozenset(instance_map.instance_to_skip)
    except (ImportError, AttributeError) as exc:
        raise AdapterInfrastructureError("official_mini_upstream_unavailable") from exc


def _expected_cases(test_spec) -> tuple[str, ...]:
    cases = (*test_spec.FAIL_TO_PASS, *test_spec.PASS_TO_PASS)
    if not test_spec.FAIL_TO_PASS or len(cases) != len(set(cases)) or any(
        not isinstance(case, str) or not case for case in cases
    ):
        raise AdapterInfrastructureError("official_mini_invalid_expected_tests")
    return cases


def grade_flask_5014(verifier_deployment, *, timeout_seconds: int = 300) -> TrustedMiniGrade:
    """Grade only the exact official Flask task in a host-private verifier.

    The caller must create a fresh verifier deployment from a frozen candidate
    patch after revoking actor access. The test patch is applied only there.
    This function returns no result if setup, execution, or log parsing fails.
    """

    Deployment, skipped = _upstream_grade_types()
    if not isinstance(verifier_deployment, Deployment):
        raise TypeError("Expected an official SandboxDeployment")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 600:
        raise ValueError("timeout_seconds must be an integer in [1, 600]")
    config = verifier_deployment._config
    ds = verifier_deployment.ds
    if (config.data_type != "swebench" or ds.get("instance_id") != INSTANCE_ID
            or ds.get("repo") != REPO or ds.get("version") != VERSION
            or ds.get("base_commit") != BASE_COMMIT or not ds.get("test_patch")):
        raise AdapterInfrastructureError("official_mini_task_binding_mismatch")
    if INSTANCE_ID in skipped:
        raise AdapterInfrastructureError("official_mini_instance_is_skip_listed")

    try:
        success, f2p, p2p, output = verifier_deployment._calculate_reward_swebench(
            get_test_output=True, timeout=timeout_seconds
        )
        test_spec = verifier_deployment.test_spec
        cases = _expected_cases(test_spec)
        if (type(success) is not bool or not isinstance(output, str)
                or output.count(START) != 1 or output.count(END) != 1
                or output.index(START) >= output.index(END)):
            raise AdapterInfrastructureError("official_mini_incomplete_test_log")
        if (test_spec.instance_id != INSTANCE_ID or test_spec.repo != REPO
                or test_spec.version != VERSION):
            raise AdapterInfrastructureError("official_mini_test_spec_mismatch")
        if (isinstance(f2p, bool) or isinstance(p2p, bool)
                or not isinstance(f2p, (int, float)) or not isinstance(p2p, (int, float))
                or not math.isfinite(f2p) or not math.isfinite(p2p)
                or not 0 <= f2p <= 1 or not 0 <= p2p <= 1):
            raise AdapterInfrastructureError("official_mini_invalid_test_ratios")
        status_map, found = verifier_deployment.get_logs_eval(test_spec, output)
        if not found or not isinstance(status_map, dict):
            raise AdapterInfrastructureError("official_mini_log_parse_failed")
        statuses = [status_map.get(case) for case in cases]
        if any(status not in PASS_STATUSES | FAIL_STATUSES for status in statuses):
            # Upstream SWE-bench grading can silently succeed when a test name
            # is absent; skipped and unknown outcomes are not evidence of pass.
            raise AdapterInfrastructureError("official_mini_expected_test_not_observed")
        resolved = all(status in PASS_STATUSES for status in statuses)
        if success != resolved:
            raise AdapterInfrastructureError("official_mini_upstream_grade_disagreement")
        f2p_count = len(test_spec.FAIL_TO_PASS)
        p2p_count = len(test_spec.PASS_TO_PASS)
        expected_f2p_ratio = sum(
            status_map[case] in PASS_STATUSES for case in test_spec.FAIL_TO_PASS
        ) / f2p_count
        expected_p2p_ratio = sum(
            status_map[case] in PASS_STATUSES for case in test_spec.PASS_TO_PASS
        ) / p2p_count if p2p_count else 1.0
        if f2p != expected_f2p_ratio or p2p != expected_p2p_ratio:
            raise AdapterInfrastructureError("official_mini_upstream_ratio_disagreement")
        return TrustedMiniGrade(
            reward=float(resolved),
            instance_id=INSTANCE_ID,
            expected_cases=len(cases),
            passed_cases=sum(status in PASS_STATUSES for status in statuses),
            failed_cases=sum(status in FAIL_STATUSES for status in statuses),
            log_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
        )
    except AdapterInfrastructureError:
        raise
    except Exception as exc:
        # Never turn installation, timeout, parser, or runtime failure into a
        # policy failure reward, as the upstream public wrapper does.
        raise AdapterInfrastructureError("official_mini_grader_infrastructure_error") from exc
