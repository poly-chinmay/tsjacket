import torch
import json
from dataclasses import dataclass
from tsjacket.compiler import CompiledSchema, compile_schema
from tsjacket.bridge import build_token_trie
from tsjacket.tracker import GrammarStateTracker
from tsjacket.zones import classify_position, ZoneType
from tsjacket.engine import apply_constraint, ConstraintDeadlockError
from tsjacket.monitor import PressureMonitor
from tsjacket.checkpoints import CheckpointManager


@dataclass
class GenerationResult:
    json_output: dict | None
    raw_string: str
    parse_success: bool
    pressure_map: dict
    rollback_count: int
    error: str | None


def generate_constrained(
    get_logits_fn,
    prompt_token_ids: list[int],
    schema: CompiledSchema,
    trie: dict[str, set[int]],
    tokenizer,
    max_new_tokens: int = 100,
    max_rollbacks: int = 3
) -> GenerationResult:

    tracker = GrammarStateTracker(schema)
    generated_ids = list(prompt_token_ids)
    output_tokens = []
    pending_token_ids: list[int] = []
    monitor = PressureMonitor()
    checkpoint_mgr = CheckpointManager()
    rollback_count = 0
    token_index = 0

    for _ in range(max_new_tokens):
        current_state = tracker.current_state
        classifier_state = current_state
        if current_state == "expect_value":
            current_field_name = getattr(tracker, "_current_field_name", None)
            if current_field_name:
                classifier_state = f"expect_value:{current_field_name}"

        if current_state == "end":
            break

        if not pending_token_ids:
            planned_literal = _planned_literal_for_state(tracker, schema)
            if planned_literal is not None:
                pending_token_ids = tokenizer.encode(planned_literal, add_special_tokens=False)

        if pending_token_ids:
            next_token_id = pending_token_ids.pop(0)

            field_name = None
            if classifier_state.startswith("expect_value:"):
                field_name = classifier_state.split("expect_value:")[1]

            if field_name is not None:
                monitor.record(
                    token_index=token_index,
                    grammar_state=classifier_state,
                    field_name=field_name,
                    pressure_score=0.0,
                )
        else:
            zone = classify_position(classifier_state, schema, trie)

            if zone.zone_type == ZoneType.STRUCTURAL:
                next_token_id = zone.forced_token_id
                if next_token_id is None:
                    break

            elif zone.zone_type == ZoneType.CONSTRAINED:
                logits = get_logits_fn(generated_ids)
                try:
                    result = apply_constraint(logits, zone.valid_token_ids, zone.zone_type)
                except ConstraintDeadlockError:
                    if rollback_count >= max_rollbacks:
                        return GenerationResult(None, "", False, {}, rollback_count, "Max rollbacks exceeded")
                    cp = checkpoint_mgr.rollback()
                    if cp is None:
                        return GenerationResult(None, "", False, {}, rollback_count, "Deadlock with no checkpoint to restore")
                    generated_ids = list(cp.input_ids_snapshot)
                    tracker.current_state = cp.grammar_state_snapshot
                    tracker.committed_fields = dict(cp.committed_fields_snapshot)
                    output_tokens = list(generated_ids[len(prompt_token_ids):])
                    token_index = cp.token_index
                    pending_token_ids = []
                    monitor.reset()
                    rollback_count += 1
                    continue

                try:
                    field_name = None
                    if classifier_state.startswith("expect_value:"):
                        field_name = classifier_state.split("expect_value:")[1]

                    monitor.record(
                        token_index=token_index,
                        grammar_state=classifier_state,
                        field_name=field_name,
                        pressure_score=result.pressure_score
                    )

                    probs = torch.softmax(result.masked_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).item()
                except ConstraintDeadlockError:
                    if rollback_count >= max_rollbacks:
                        return GenerationResult(None, "", False, {}, rollback_count, "Max rollbacks exceeded")
                    cp = checkpoint_mgr.rollback()
                    if cp is None:
                        return GenerationResult(None, "", False, {}, rollback_count, "Deadlock with no checkpoint to restore")
                    generated_ids = list(cp.input_ids_snapshot)
                    tracker.current_state = cp.grammar_state_snapshot
                    tracker.committed_fields = dict(cp.committed_fields_snapshot)
                    output_tokens = list(generated_ids[len(prompt_token_ids):])
                    token_index = cp.token_index
                    pending_token_ids = []
                    monitor.reset()
                    rollback_count += 1
                    continue

            else:  # SEMANTIC
                logits = get_logits_fn(generated_ids)
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                field_name = None
                if current_state.startswith("expect_value:"):
                    field_name = current_state.split("expect_value:")[1]
                elif classifier_state.startswith("expect_value:"):
                    field_name = classifier_state.split("expect_value:")[1]

                monitor.record(
                    token_index=token_index,
                    grammar_state=current_state,
                    field_name=field_name,
                    pressure_score=0.0
                )

        token_str = tokenizer.decode([next_token_id])
        advance_result = tracker.advance(token_str)
        generated_ids.append(next_token_id)
        output_tokens.append(next_token_id)

        if advance_result.field_boundary_crossed:
            checkpoint_mgr.save(
                field_index=len(tracker.committed_fields) - 1,
                field_name=advance_result.completed_field_name,
                token_index=token_index,
                input_ids=list(generated_ids),
                grammar_state=tracker.current_state,
                committed_fields=dict(tracker.committed_fields)
            )

        token_index += 1

        if tracker.current_state == "end":
            break

    raw = tokenizer.decode(output_tokens)
    if tracker.current_state == "end":
        stripped_raw = raw.rstrip()
        if stripped_raw.startswith("{") and not stripped_raw.endswith("}"):
            raw = f"{raw}}}"
    pressure_map = monitor.format_output()

    try:
        parsed = json.loads(raw)
        return GenerationResult(parsed, raw, True, pressure_map, rollback_count, None)
    except json.JSONDecodeError as e:
        return GenerationResult(None, raw, False, pressure_map, rollback_count, str(e))


def _planned_literal_for_state(tracker: GrammarStateTracker, schema: CompiledSchema) -> str | None:
    state = tracker.current_state

    if state == "start":
        return "{"

    if state == "expect_key":
        field_name = getattr(tracker, "_current_field_name", None)
        if field_name is None:
            return None
        return f'"{field_name}"'

    if state == "expect_colon":
        return ":"

    if state == "expect_separator":
        return ","

    if state == "expect_value":
        field_name = getattr(tracker, "_current_field_name", None)
        if field_name is None:
            return None

        enums = schema.field_enums.get(field_name)
        if enums:
            return f'"{enums[0]}"'

        field_type = schema.field_types.get(field_name)
        if field_type == "string":
            return '"x"'
        if field_type == "integer":
            return "1"
        if field_type == "number":
            return "1"
        if field_type == "boolean":
            return "true"

    return None


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("./test_tokenizer")
    vocab_size = tokenizer.vocab_size

    schema = compile_schema({
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["active", "inactive"]},
            "count": {"type": "integer"}
        },
        "required": ["status", "count"]
    })
    trie = build_token_trie(schema, tokenizer)

    def mock_get_logits(input_ids: list[int]) -> torch.Tensor:
        torch.manual_seed(len(input_ids))
        return torch.randn(vocab_size)

    prompt_ids = tokenizer.encode(
        'Generate a JSON object with status and count fields:',
        add_special_tokens=False
    )

    result = generate_constrained(
        mock_get_logits,
        prompt_ids,
        schema,
        trie,
        tokenizer,
        max_new_tokens=80
    )

    print(f"Raw output: {result.raw_string!r}")
    print(f"Parse success: {result.parse_success}")
    print(f"Pressure map: {json.dumps(result.pressure_map, indent=2)}")
    print(f"Rollback count: {result.rollback_count}")
    print(f"Error: {result.error}")

    if result.parse_success:
        print(f"Parsed JSON: {result.json_output}")
        assert "status" in result.json_output
        assert result.json_output["status"] in ["active", "inactive"]
        for fname in result.pressure_map:
            assert fname in schema.fields, f"Unknown field in pressure map: {fname}"
        print("generator: ALL TESTS PASSED")
    else:
        print("generator: FAILED — check tracker alignment")
