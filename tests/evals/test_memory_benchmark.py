from __future__ import annotations

import json
import re
import shlex
import sys
from collections import Counter
from importlib.metadata import distributions
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from evals.memory_benchmark.adapters import (
    AppendOnlyTranscriptAdapter,
    CommandAdapter,
    HexisMemoryAdapter,
    RecentWindowAdapter,
)
from evals.memory_benchmark.model import (
    DIMENSIONS,
    EXPECTED_DATASET_SHA256,
    Prediction,
    dataset_path,
    dataset_sha256,
    load_cases,
)
from evals.memory_benchmark.scoring import score_case, score_predictions
from evals.memory_benchmark.run import _version_state


def test_public_corpus_is_balanced_versioned_and_schema_valid() -> None:
    cases = load_cases()
    assert len(cases) == 25
    assert Counter(case.dimension for case in cases) == {
        dimension: 5 for dimension in DIMENSIONS
    }
    assert dataset_sha256() == EXPECTED_DATASET_SHA256

    schema = json.loads(
        dataset_path().with_name("case.schema.v1.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for line in dataset_path().read_text(encoding="utf-8").splitlines():
        validator.validate(json.loads(line))


def test_version_state_does_not_mistake_checkout_metadata_for_install() -> None:
    source_root = Path(__file__).resolve().parents[2]
    project_text = (source_root / "pyproject.toml").read_text(encoding="utf-8")
    source_match = re.search(
        r"(?ms)^\[project\]\s*$.*?^version\s*=\s*[\"']([^\"']+)[\"']",
        project_text,
    )
    assert source_match is not None
    installed = next(
        candidate.version
        for candidate in distributions(name="hexis")
        if Path(candidate.locate_file("")).resolve() != source_root
    )

    state = _version_state()

    assert state["hexis_source_version"] == source_match.group(1)
    assert state["installed_distribution_version"] == installed


def test_adapter_input_never_contains_gold_answers() -> None:
    public = load_cases()[0].public_dict()
    encoded = json.dumps(public)
    assert "expected" not in public
    assert "LARCH-742" in encoded  # the statement is observable, as it must be
    assert "forbidden_answers" not in encoded
    schema = json.loads(
        dataset_path()
        .with_name("adapter-input.schema.v1.json")
        .read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(public)


def test_scorer_rewards_exact_evidence_and_penalizes_stale_output() -> None:
    cases = {case.case_id: case for case in load_cases()}
    provenance = cases["provenance-01"]
    perfect = score_case(
        provenance,
        Prediction(
            case_id=provenance.case_id,
            answer="The code is LARCH-742.",
            citations=("p01-user-note",),
        ),
    )
    over_cited = score_case(
        provenance,
        Prediction(
            case_id=provenance.case_id,
            answer="The code is LARCH-742.",
            citations=("p01-user-note", "p01-distractor"),
        ),
    )
    stale = cases["stale-01"]
    stale_output = score_case(
        stale,
        Prediction(
            case_id=stale.case_id,
            answer="It changed from NORTHSTAR-OLD to BLUEHARBOR-NEW.",
            citations=("s01-old", "s01-new"),
        ),
    )

    assert perfect["score"] == 100
    assert over_cited["score"] < perfect["score"]
    assert stale_output["score"] == 50
    assert stale_output["forbidden_matches"] == ["NORTHSTAR-OLD"]


def test_published_hexis_submission_reproduces_the_published_score() -> None:
    results_dir = dataset_path().with_name("results")
    summary = json.loads((results_dir / "2026-08-28.json").read_text(encoding="utf-8"))
    raw_predictions = json.loads(
        (results_dir / "2026-08-28-hexis-predictions.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        dataset_path()
        .with_name("prediction.schema.v1.json")
        .read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema)
    for prediction in raw_predictions:
        validator.validate(prediction)
    scored = score_predictions(
        load_cases(),
        [Prediction.from_dict(item) for item in raw_predictions],
    )
    published = next(
        system for system in summary["systems"] if system["name"] == "hexis-memory-v1"
    )

    assert scored["overall"] == published["overall"] == 96.33
    assert {
        dimension: detail["score"] for dimension, detail in scored["dimensions"].items()
    } == published["dimensions"]


@pytest.mark.asyncio(loop_scope="session")
async def test_reference_baselines_lock_expected_sanity_profile() -> None:
    cases = load_cases()
    append_predictions = [
        await AppendOnlyTranscriptAdapter().predict(case) for case in cases
    ]
    recent_predictions = [await RecentWindowAdapter().predict(case) for case in cases]

    append = score_predictions(cases, append_predictions)
    recent = score_predictions(cases, recent_predictions)

    assert append["overall"] == 82.33
    assert append["dimensions"]["stale_belief_resistance"]["score"] == 60
    assert recent["overall"] == 32
    assert recent["dimensions"]["six_month_recall"]["score"] == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_command_adapter_receives_gold_free_case_and_validates_prediction(
    tmp_path,
) -> None:
    wrapper = tmp_path / "adapter.py"
    wrapper.write_text(
        """
import json
import sys
case = json.load(sys.stdin)
assert "expected" not in case
event = case["events"][0]
json.dump({
    "case_id": case["case_id"],
    "answer": event["content"],
    "citations": [event["event_id"]],
    "contradictions": [],
    "abstained": False,
}, sys.stdout)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(wrapper))}"
    adapter = CommandAdapter(command, name="test-command", timeout_seconds=10)

    prediction = await adapter.predict(load_cases()[0])

    assert prediction.abstained is False
    assert prediction.citations == ("p01-user-note",)
    assert "LARCH-742" in prediction.answer
    schema = json.loads(
        dataset_path()
        .with_name("prediction.schema.v1.json")
        .read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(prediction.as_dict())


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.db
async def test_hexis_adapter_uses_real_memory_functions_and_rolls_back(db_pool) -> None:
    cases = {case.case_id: case for case in load_cases()}
    selected = [
        cases["provenance-01"],
        cases["six-month-01"],
        cases["continuity-01"],
        cases["stale-01"],
    ]
    async with db_pool.acquire() as conn:
        before = await conn.fetchval(
            "SELECT count(*) FROM memories WHERE metadata ? 'benchmark_case_id'"
        )

    adapter = HexisMemoryAdapter(db_pool, live_contradictions=False)
    predictions = [await adapter.predict(case) for case in selected]

    for case, prediction in zip(selected, predictions, strict=True):
        assert prediction.abstained is False, prediction.metadata
        for answer in case.expected.answers:
            assert answer in prediction.answer
    assert "NORTHSTAR-OLD" not in predictions[-1].answer
    assert predictions[-1].citations == ("s01-new",)
    async with db_pool.acquire() as conn:
        after = await conn.fetchval(
            "SELECT count(*) FROM memories WHERE metadata ? 'benchmark_case_id'"
        )
    assert after == before
