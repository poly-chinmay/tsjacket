from tsjacket.compiler import CompiledSchema


def build_token_trie(schema: CompiledSchema, tokenizer) -> dict[str, set[int]]:
    """
    Returns: dict mapping grammar_state_name -> set of valid next token IDs

    Grammar states to handle:
    - "start"                   → valid strings: ["{"]
    - "expect_key"              → valid strings: ['"field_name"' for each field]
    - "expect_colon"            → valid strings: [":"]
    - "expect_value:{field}"    → depends on field type (see below)
    - "expect_separator"        → valid strings: [",", "}"]
    - "end"                     → empty set
    """
    trie = {}

    def first_tokens_of(strings: list[str]) -> set[int]:
        valid_ids = set()
        for s in strings:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if ids:
                valid_ids.add(ids[0])
        return valid_ids

    trie["start"] = first_tokens_of(["{"])

    field_name_strings = [f'"{f}"' for f in schema.fields]
    trie["expect_key"] = first_tokens_of(field_name_strings)

    trie["expect_colon"] = first_tokens_of([":", " :"])

    trie["expect_separator"] = first_tokens_of([",", " ,", "}", " }"])

    for field_name in schema.fields:
        state_key = f"expect_value:{field_name}"
        ftype = schema.field_types[field_name]
        enums = schema.field_enums.get(field_name)

        if enums is not None:
            valid_strings = [f'"{v}"' for v in enums]
        elif ftype == "string":
            valid_strings = ['"']
        elif ftype in ("integer", "number"):
            valid_strings = [str(d) for d in range(10)] + ["-"]
        elif ftype == "boolean":
            valid_strings = ["true", "false"]
        else:
            valid_strings = []

        trie[state_key] = first_tokens_of(valid_strings)

    trie["end"] = set()
    return trie


def tokens_for_values(values: list, tokenizer) -> set[int]:
    """
    Given a list of allowed values, return the set of valid
    first token IDs. Used by constraint graph to override
    trie token sets at generation time.
    """
    valid_ids = set()
    for v in values:
        # Handle bool, int, float, str
        if isinstance(v, bool):
            s = "true" if v else "false"
        else:
            s = f'"{v}"' if isinstance(v, str) else str(v)
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            valid_ids.add(ids[0])
    return valid_ids


if __name__ == "__main__":
    from tsjacket.compiler import compile_schema
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("./test_tokenizer")

    schema = compile_schema({
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "population": {"type": "integer"},
            "active": {"type": "boolean"},
            "status": {"type": "string", "enum": ["active", "inactive"]}
        },
        "required": ["city", "population", "active", "status"]
    })

    trie = build_token_trie(schema, tokenizer)

    open_brace_id = tokenizer.encode("{", add_special_tokens=False)[0]
    assert open_brace_id in trie["start"]

    true_id = tokenizer.encode("true", add_special_tokens=False)[0]
    assert true_id in trie["expect_value:active"]

    status_ids = trie["expect_value:status"]
    assert len(status_ids) > 0

    digit_id = tokenizer.encode("1", add_special_tokens=False)[0]
    assert digit_id in trie["expect_value:population"]

    print("States built:", list(trie.keys()))
    print("Token counts per state:", {k: len(v) for k, v in trie.items()})
    print("bridge: ALL TESTS PASSED")
