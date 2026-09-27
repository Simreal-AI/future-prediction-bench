"""Humanize 4.15.0 opt-in binding for the cooperative graded environment."""

from __future__ import annotations

from pathlib import Path

from examples.cooperative_realworld.host_runtime import (
    CooperativePrivateCases, CooperativeRealWorldAdapter, CooperativeClient,
    connect_booted_vm, verify_assets,
)
from examples.realworld_humanize.make_task import VISIBLE_CHECK
from future_prediction_bench.realworld import AdapterInfrastructureError, RealWorldEnv

HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "cooperative_contract.json"
RESIDENT_CONTRACT = HERE / "resident_contract.json"


class HumanizePrivateCases(CooperativePrivateCases):
    def __init__(self, task_path, verifier_path, assets_manifest_path,
                 *, task_window=None):
        super().__init__(task_path, verifier_path, assets_manifest_path,
                         contract_path=CONTRACT,
                         resident_contract_path=RESIDENT_CONTRACT,
                         task_window=task_window)
        if (self.task_id != "humanize-4150-naturalsize-rounding-v2"
                or self.source_path != "src/humanize/filesize.py"):
            raise ValueError("humanize_cooperative_fixture_mismatch")
        self.visible_check = VISIBLE_CHECK


def make_env(client: CooperativeClient, private: HumanizePrivateCases,
             *, mode, clock=None):
    if not isinstance(private, HumanizePrivateCases):
        raise TypeError("humanize_private_cases_required")
    if not isinstance(client, CooperativeClient) or client.state != "idle":
        raise ValueError("connected_idle_cooperative_client_required")
    adapter = CooperativeRealWorldAdapter(
        client, private, infrastructure_error_class=AdapterInfrastructureError,
        episode_mode=mode, case_transport="batch")
    return RealWorldEnv(private.experimental_task(), adapter, clock=clock)
