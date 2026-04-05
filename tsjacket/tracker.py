from dataclasses import dataclass, field
from tsjacket.compiler import CompiledSchema


@dataclass
class AdvanceResult:
    next_state: str
    field_boundary_crossed: bool
    completed_field_name: str | None
    completed_field_value: str | None


class GrammarStateTracker:
    def __init__(self, schema: CompiledSchema):
        self.schema = schema
        self.current_state = "start"
        self.committed_fields: dict[str, any] = {}
        self._current_field_idx = -1
        self._current_field_name: str | None = None
        self._accumulating_value = False
        self._value_buffer = ""
        self._in_string = False
        self._string_complete = False

    def advance(self, token_str: str) -> AdvanceResult:
        crossed = False
        completed_name = None
        completed_value = None

        if self.current_state == "start":
            if token_str.strip() == "{":
                self._current_field_idx = 0
                self._current_field_name = self.schema.fields[0]
                self.current_state = "expect_key"

        elif self.current_state == "expect_key":
            self.current_state = "expect_colon"

        elif self.current_state == "expect_colon":
            if ":" in token_str:
                self.current_state = "expect_value"
                self._value_buffer = ""
                self._in_string = False
                self._string_complete = False

        elif self.current_state == "expect_value":
            ftype = self.schema.field_types[self._current_field_name]
            self._value_buffer += token_str

            if ftype == "string" or (self.schema.field_enums.get(self._current_field_name)):
                stripped = self._value_buffer.strip()
                if stripped.startswith('"') and stripped.endswith('"') and len(stripped) >= 2:
                    self._commit_field()
                    crossed = True
                    completed_name = self._current_field_name
                    completed_value = stripped.strip('"')
                    self._advance_to_next_field()
            elif ftype in ("integer", "number", "boolean"):
                stripped = self._value_buffer.strip()
                if stripped in ("true", "false") or self._is_complete_number(stripped):
                    self._commit_field()
                    crossed = True
                    completed_name = self._current_field_name
                    completed_value = stripped
                    self._advance_to_next_field()

        elif self.current_state == "expect_separator":
            if "," in token_str:
                self.current_state = "expect_key"
            elif "}" in token_str:
                self.current_state = "end"

        return AdvanceResult(self.current_state, crossed, completed_name, completed_value)

    def _is_complete_number(self, s: str) -> bool:
        try:
            float(s.strip())
            return True
        except:
            return False

    def _commit_field(self):
        name = self._current_field_name
        val = self._value_buffer.strip().strip('"')
        ftype = self.schema.field_types[name]
        if ftype in ("integer",):
            try:
                val = int(val)
            except:
                pass
        elif ftype == "number":
            try:
                val = float(val)
            except:
                pass
        self.committed_fields[name] = val

    def _advance_to_next_field(self):
        self._current_field_idx += 1
        if self._current_field_idx >= len(self.schema.fields):
            self.current_state = "end"
        else:
            self._current_field_name = self.schema.fields[self._current_field_idx]
            self.current_state = "expect_separator"


if __name__ == "__main__":
    from tsjacket.compiler import compile_schema

    schema = compile_schema({
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "population": {"type": "integer"}
        },
        "required": ["city", "population"]
    })
    tracker = GrammarStateTracker(schema)
    tokens = ["{", '"city"', ":", '"Paris"', ",", '"population"', ":", "2161000", "}"]
    for t in tokens:
        r = tracker.advance(t)
        print(f"token={t!r:20s} state={r.next_state:25s} crossed={r.field_boundary_crossed}")
    assert tracker.committed_fields.get("city") == "Paris"
    assert tracker.committed_fields.get("population") == 2161000
    print("tracker: ALL TESTS PASSED")
