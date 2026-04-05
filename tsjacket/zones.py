from dataclasses import dataclass
from enum import Enum
from tsjacket.compiler import CompiledSchema


class ZoneType(Enum):
    STRUCTURAL = "structural"
    CONSTRAINED = "constrained"
    SEMANTIC = "semantic"


@dataclass
class ZoneInfo:
    zone_type: ZoneType
    valid_token_ids: set[int] | None
    forced_token_id: int | None


SEMANTIC_THRESHOLD = 50


def classify_position(
    grammar_state: str,
    schema: CompiledSchema,
    trie: dict[str, set[int]]
) -> ZoneInfo:

    valid_ids = trie.get(grammar_state, set())
    count = len(valid_ids)

    if count == 0:
        return ZoneInfo(ZoneType.STRUCTURAL, set(), None)

    if count == 1:
        forced = next(iter(valid_ids))
        return ZoneInfo(ZoneType.STRUCTURAL, valid_ids, forced)

    if grammar_state.startswith("expect_value:"):
        field_name = grammar_state.split("expect_value:")[1]
        ftype = schema.field_types.get(field_name)
        enums = schema.field_enums.get(field_name)
        if ftype == "string" and enums is None:
            return ZoneInfo(ZoneType.SEMANTIC, None, None)

    if count > SEMANTIC_THRESHOLD:
        return ZoneInfo(ZoneType.SEMANTIC, None, None)

    return ZoneInfo(ZoneType.CONSTRAINED, valid_ids, None)


if __name__ == "__main__":
    from tsjacket.compiler import compile_schema
    from tsjacket.bridge import build_token_trie
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("./test_tokenizer")
    schema = compile_schema({
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "population": {"type": "integer"},
            "status": {"type": "string", "enum": ["active", "inactive"]}
        },
        "required": ["city", "population", "status"]
    })
    trie = build_token_trie(schema, tokenizer)

    info = classify_position("start", schema, trie)
    assert info.zone_type == ZoneType.STRUCTURAL

    info = classify_position("expect_value:city", schema, trie)
    assert info.zone_type == ZoneType.SEMANTIC

    info = classify_position("expect_value:status", schema, trie)
    assert info.zone_type == ZoneType.CONSTRAINED

    info = classify_position("expect_value:population", schema, trie)
    assert info.zone_type == ZoneType.CONSTRAINED

    print("zones: ALL TESTS PASSED")
