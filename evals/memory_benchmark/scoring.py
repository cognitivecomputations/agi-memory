"""Deterministic, judge-free scoring for memory benchmark submissions."""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import defaultdict
from typing import Any, Iterable

from .model import BENCHMARK_NAME, BENCHMARK_VERSION, BenchmarkCase, Prediction


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[\w]+", value, flags=re.UNICODE))


def _contains(answer: str, expected: str) -> bool:
    needle = _normalized(expected)
    haystack = _normalized(answer)
    return bool(needle and re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack))


def _set_f1(predicted: Iterable[str], expected: Iterable[str]) -> float:
    predicted_set = set(predicted)
    expected_set = set(expected)
    if not predicted_set and not expected_set:
        return 1.0
    if not predicted_set or not expected_set:
        return 0.0
    true_positive = len(predicted_set & expected_set)
    precision = true_positive / len(predicted_set)
    recall = true_positive / len(expected_set)
    return (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )


def score_case(case: BenchmarkCase, prediction: Prediction) -> dict[str, Any]:
    answer_matches = [
        answer
        for answer in case.expected.answers
        if _contains(prediction.answer, answer)
    ]
    forbidden_matches = [
        answer
        for answer in case.expected.forbidden_answers
        if _contains(prediction.answer, answer)
    ]
    expected_recall = len(answer_matches) / len(case.expected.answers)
    forbidden_clean = (
        1.0
        if not case.expected.forbidden_answers
        else 1.0 - len(forbidden_matches) / len(case.expected.forbidden_answers)
    )
    answer_score = expected_recall * forbidden_clean
    citation_f1 = _set_f1(prediction.citations, case.expected.citations)
    contradiction_f1 = _set_f1(
        prediction.contradictions,
        case.expected.contradictions,
    )

    if case.dimension == "provenance_accuracy":
        score = 0.5 * answer_score + 0.5 * citation_f1
    elif case.dimension == "contradiction_detection":
        score = 0.5 * answer_score + 0.5 * contradiction_f1
    elif case.dimension == "stale_belief_resistance":
        score = 0.5 * expected_recall + 0.5 * forbidden_clean
    else:
        score = answer_score
    if prediction.abstained:
        score = 0.0

    return {
        "case_id": case.case_id,
        "dimension": case.dimension,
        "score": round(score * 100, 2),
        "answer_recall": round(expected_recall * 100, 2),
        "forbidden_clean": round(forbidden_clean * 100, 2),
        "citation_f1": round(citation_f1 * 100, 2),
        "contradiction_f1": round(contradiction_f1 * 100, 2),
        "matched_answers": answer_matches,
        "forbidden_matches": forbidden_matches,
        "abstained": prediction.abstained,
    }


def score_predictions(
    cases: list[BenchmarkCase],
    predictions: list[Prediction],
) -> dict[str, Any]:
    case_by_id = {case.case_id: case for case in cases}
    prediction_by_id: dict[str, Prediction] = {}
    unknown: list[str] = []
    duplicates: list[str] = []
    for prediction in predictions:
        if prediction.case_id not in case_by_id:
            unknown.append(prediction.case_id)
            continue
        if prediction.case_id in prediction_by_id:
            duplicates.append(prediction.case_id)
            continue
        prediction_by_id[prediction.case_id] = prediction
    if unknown:
        raise ValueError(
            f"predictions contain unknown case ids: {sorted(set(unknown))}"
        )
    if duplicates:
        raise ValueError(
            f"predictions contain duplicate case ids: {sorted(set(duplicates))}"
        )

    case_scores: list[dict[str, Any]] = []
    for case in cases:
        prediction = prediction_by_id.get(
            case.case_id,
            Prediction(case_id=case.case_id, answer="", abstained=True),
        )
        case_scores.append(score_case(case, prediction))

    by_dimension: dict[str, list[float]] = defaultdict(list)
    for result in case_scores:
        by_dimension[result["dimension"]].append(float(result["score"]))
    dimensions = {
        dimension: {
            "score": round(statistics.fmean(values), 2),
            "cases": len(values),
        }
        for dimension, values in by_dimension.items()
    }
    overall = statistics.fmean(item["score"] for item in dimensions.values())
    missing = sorted(set(case_by_id) - set(prediction_by_id))
    return {
        "benchmark": BENCHMARK_NAME,
        "version": BENCHMARK_VERSION,
        "overall": round(overall, 2),
        "dimensions": dimensions,
        "case_count": len(cases),
        "submitted_count": len(prediction_by_id),
        "missing_case_ids": missing,
        "cases": case_scores,
    }
