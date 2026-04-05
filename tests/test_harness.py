from dataclasses import dataclass
import json
import time
from typing import Any


STRESS_PROMPT_EXAMPLES = [
    "Give me a JSON where age is 'twenty five'",
    "Return JSON with extra fields not in schema",
    "Output JSON where status is 'pending'",
]


@dataclass
class TestCase:
    prompt: str
    schema: dict
    expect_fields: list[str]
    expect_types: dict[str, type]
    stress: bool


@dataclass
class TestResult:
    case: TestCase
    output_json: dict | None
    parse_success: bool
    type_correct: bool
    latency_ms: float
    pressure_map: dict
    rollback_count: int
    error: str | None


def run_test_suite(cases: list[TestCase], model, tokenizer) -> list[TestResult]:
    results: list[TestResult] = []
    total_rollbacks = 0
    total_latency_ms = 0.0
    total_output_tokens = 0
    flagged_fields: dict[str, int] = {}

    for case in cases:
        start = time.perf_counter()
        output_json: dict | None = None
        parse_success = False
        type_correct = False
        pressure_map: dict = {}
        rollback_count = 0
        error: str | None = None
        output_token_count = 0

        try:
            generated = _invoke_generation(case, model, tokenizer)

            output_json = generated.get("output_json")
            pressure_map = generated.get("pressure_map") or {}
            rollback_count = int(generated.get("rollback_count") or 0)
            output_token_count = int(generated.get("output_token_count") or 0)

            parse_success = _is_structurally_valid(output_json, case.expect_fields)
            type_correct = parse_success and _types_match(output_json, case.expect_types)

        except Exception as exc:
            error = str(exc)

        latency_ms = (time.perf_counter() - start) * 1000.0

        result = TestResult(
            case=case,
            output_json=output_json,
            parse_success=parse_success,
            type_correct=type_correct,
            latency_ms=latency_ms,
            pressure_map=pressure_map,
            rollback_count=rollback_count,
            error=error,
        )
        results.append(result)

        total_rollbacks += rollback_count
        total_latency_ms += latency_ms
        total_output_tokens += max(0, output_token_count)
        _collect_flagged_fields(pressure_map, flagged_fields)

    _print_results_table(results)
    _print_metrics(
        results=results,
        total_latency_ms=total_latency_ms,
        total_output_tokens=total_output_tokens,
        flagged_fields=flagged_fields,
        total_rollbacks=total_rollbacks,
    )

    return results


def _invoke_generation(case: TestCase, model, tokenizer) -> dict[str, Any]:
    if hasattr(model, "generate_constrained"):
        payload = model.generate_constrained(prompt=case.prompt, schema=case.schema, tokenizer=tokenizer)
        return _normalize_payload(payload, tokenizer)

    if hasattr(model, "constrained_generate"):
        payload = model.constrained_generate(prompt=case.prompt, schema=case.schema, tokenizer=tokenizer)
        return _normalize_payload(payload, tokenizer)

    encoded = tokenizer(case.prompt, return_tensors="pt")
    output_ids = model.generate(**encoded, max_new_tokens=128)

    if hasattr(output_ids, "shape") and len(output_ids.shape) == 2:
        generated_ids = output_ids[0]
    else:
        generated_ids = output_ids

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    parsed_json = _try_parse_json(output_text)
    token_count = int(getattr(generated_ids, "shape", [0])[0]) if hasattr(generated_ids, "shape") else 0

    return {
        "output_json": parsed_json,
        "pressure_map": {},
        "rollback_count": 0,
        "output_token_count": token_count,
    }


def _normalize_payload(payload: Any, tokenizer) -> dict[str, Any]:
    if isinstance(payload, dict):
        output_json = payload.get("output_json")
        output_text = payload.get("output_text")

        if output_json is None and isinstance(output_text, str):
            output_json = _try_parse_json(output_text)

        token_count = payload.get("output_token_count")
        if token_count is None and isinstance(output_text, str):
            token_count = len(tokenizer.encode(output_text)) if tokenizer is not None else 0

        return {
            "output_json": output_json,
            "pressure_map": payload.get("pressure_map") or {},
            "rollback_count": int(payload.get("rollback_count") or 0),
            "output_token_count": int(token_count or 0),
        }

    if isinstance(payload, str):
        return {
            "output_json": _try_parse_json(payload),
            "pressure_map": {},
            "rollback_count": 0,
            "output_token_count": len(tokenizer.encode(payload)) if tokenizer is not None else 0,
        }

    raise TypeError("Unsupported generation payload from model")


def _try_parse_json(text: str) -> dict | None:
    block = _extract_first_json_object(text)
    if block is None:
        return None

    try:
        parsed = json.loads(block)
    except Exception:
        return None

    return parsed if isinstance(parsed, dict) else None


def _extract_first_json_object(text: str) -> str | None:
    depth = 0
    start_idx: int | None = None
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
            continue

        if ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_idx is not None:
                return text[start_idx : i + 1]

    return None


def _is_structurally_valid(output_json: dict | None, expect_fields: list[str]) -> bool:
    if output_json is None:
        return False

    expected = set(expect_fields)
    actual = set(output_json.keys())
    return actual == expected


def _types_match(output_json: dict | None, expect_types: dict[str, type]) -> bool:
    if output_json is None:
        return False

    for field_name, expected_type in expect_types.items():
        if field_name not in output_json:
            return False
        value = output_json[field_name]
        if not isinstance(value, expected_type):
            return False
    return True


def _collect_flagged_fields(pressure_map: dict, aggregate: dict[str, int]) -> None:
    for field_name, info in pressure_map.items():
        flagged = False

        if isinstance(info, dict):
            flagged = bool(info.get("flagged", False))
        elif hasattr(info, "flagged"):
            flagged = bool(getattr(info, "flagged"))

        if flagged:
            aggregate[field_name] = aggregate.get(field_name, 0) + 1


def _print_results_table(results: list[TestResult]) -> None:
    headers = [
        "idx",
        "stress",
        "parse",
        "type",
        "latency_ms",
        "rollbacks",
        "error",
        "prompt",
    ]

    rows: list[list[str]] = []
    for idx, res in enumerate(results):
        rows.append(
            [
                str(idx),
                "Y" if res.case.stress else "N",
                "Y" if res.parse_success else "N",
                "Y" if res.type_correct else "N",
                f"{res.latency_ms:.2f}",
                str(res.rollback_count),
                (res.error or "")[:40],
                res.case.prompt[:50],
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if len(cell) > widths[i]:
                widths[i] = len(cell)

    def _fmt_row(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    divider = "-+-".join("-" * width for width in widths)

    print(_fmt_row(headers))
    print(divider)
    for row in rows:
        print(_fmt_row(row))


def _print_metrics(
    results: list[TestResult],
    total_latency_ms: float,
    total_output_tokens: int,
    flagged_fields: dict[str, int],
    total_rollbacks: int,
) -> None:
    total = len(results)
    if total == 0:
        print("No test cases provided.")
        return

    structural_validity_rate = sum(1 for r in results if r.parse_success) / total
    type_correctness_rate = sum(1 for r in results if r.type_correct) / total

    mean_latency_per_token = 0.0
    if total_output_tokens > 0:
        mean_latency_per_token = total_latency_ms / total_output_tokens

    flagged_summary = sorted(flagged_fields.items(), key=lambda item: (-item[1], item[0]))

    print()
    print(f"Structural validity rate: {structural_validity_rate * 100:.2f}%")
    print(f"Type correctness rate: {type_correctness_rate * 100:.2f}%")
    print(f"Mean latency per token: {mean_latency_per_token:.4f} ms")
    print(f"Total rollbacks: {total_rollbacks}")

    if flagged_summary:
        print("Fields flagged by pressure monitor:")
        for field_name, count in flagged_summary:
            print(f"  - {field_name}: {count}")
    else:
        print("Fields flagged by pressure monitor: none")
