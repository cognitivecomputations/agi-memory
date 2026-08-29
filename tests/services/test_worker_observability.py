from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from services import worker_service
from services.worker_service import MaintenanceWorker, _result_has_work


def test_result_has_work_filters_idle_poll_results():
    assert _result_has_work(None) is False
    assert _result_has_work(0) is False
    assert _result_has_work([]) is False
    assert _result_has_work({"skipped": True, "reason": "idle"}) is False

    assert _result_has_work(1) is True
    assert _result_has_work(["claimed"]) is True
    assert _result_has_work({"claimed": 1}) is True
    assert _result_has_work({"skipped": False, "processed": 0}) is True


def test_contradiction_tasks_follow_embedding_and_are_both_registered():
    names = [name for name, _ in MaintenanceWorker()._maintenance_task_runners()]

    assert names.index("memory_embedding") < names.index("contradiction_detection")
    assert names.index("contradiction_detection") < names.index("contradiction_digest")
    assert names.count("contradiction_detection") == 1
    assert names.count("contradiction_digest") == 1


@pytest.mark.asyncio
async def test_reported_failure_is_not_recorded_as_completed(monkeypatch):
    record = AsyncMock(return_value="run-1")
    monkeypatch.setattr(worker_service, "_record_worker_task_outcome", record)
    worker = MaintenanceWorker()
    worker.pool = object()
    worker.worker_id = "11111111-1111-4111-8111-111111111111"

    result = await worker._run_observed_task(
        "contradiction_detection",
        AsyncMock(return_value={"failed": True, "error": "provider unavailable"}),
    )

    assert result == {"failed": True, "error": "provider unavailable"}
    assert record.await_args.kwargs["status"] == "failed"
    assert record.await_args.kwargs["error"] == "provider unavailable"
