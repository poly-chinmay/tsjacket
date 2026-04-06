from dataclasses import dataclass, field
from typing import Any


@dataclass
class FieldConstraint:
    if_field: str
    if_value: Any
    then_field: str
    then_value: Any


@dataclass
class ConstraintGraphState:
    committed: dict = field(default_factory=dict)
    restrictions: dict = field(default_factory=dict)
    violations: list = field(default_factory=list)


class ConstraintGraph:
    def __init__(self, rules: list):
        self._constraints = self._parse(rules)

    def _parse(self, rules: list) -> list[FieldConstraint]:
        parsed = []
        for rule in rules:
            if isinstance(rule, dict):
                parsed.append(FieldConstraint(
                    if_field=rule["if"]["field"],
                    if_value=rule["if"]["eq"],
                    then_field=rule["then"]["field"],
                    then_value=rule["then"]["must_be"]
                ))
            elif isinstance(rule, str):
                parsed.append(self._parse_string_rule(rule))
        return parsed

    def _parse_string_rule(self, rule: str) -> FieldConstraint:
        # Format: "if FIELD == VALUE then FIELD == VALUE"
        import re
        pattern = r"if (\w+)\s*==\s*(\S+)\s+then (\w+)\s*==\s*(\S+)"
        m = re.match(pattern, rule.strip())
        if not m:
            raise ValueError(f"Cannot parse constraint rule: {rule!r}")
        if_field, if_val, then_field, then_val = m.groups()
        # Type coerce then_value
        then_val = self._coerce(then_val)
        if_val = self._coerce(if_val)
        return FieldConstraint(if_field, if_val, then_field, then_val)

    def _coerce(self, val: str) -> Any:
        if val.lower() == "true":
            return True
        if val.lower() == "false":
            return False
        try:
            return int(val)
        except:
            pass
        try:
            return float(val)
        except:
            pass
        return val.strip('"').strip("'")

    def fresh_state(self) -> ConstraintGraphState:
        return ConstraintGraphState()

    def commit_field(self, field_name: str, value: Any,
                     state: ConstraintGraphState) -> ConstraintGraphState:
        state.committed[field_name] = value
        for c in self._constraints:
            if c.if_field != field_name:
                continue
            if self._matches(value, c.if_value):
                if c.then_field in state.restrictions:
                    existing = state.restrictions[c.then_field]
                    if c.then_value not in existing:
                        state.violations.append(
                            f"Conflict: {field_name}={value!r} requires "
                            f"{c.then_field}={c.then_value!r} but "
                            f"restriction already set to {existing}"
                        )
                        continue
                state.restrictions[c.then_field] = {c.then_value}
        return state

    def get_allowed_values(self, field_name: str,
                           schema_enum: list | None,
                           state: ConstraintGraphState) -> list | None:
        if field_name not in state.restrictions:
            return schema_enum
        restriction = state.restrictions[field_name]
        if schema_enum is not None:
            allowed = [v for v in schema_enum if v in restriction]
        else:
            allowed = list(restriction)
        return allowed

    def is_satisfiable(self, field_name: str,
                       state: ConstraintGraphState) -> bool:
        if field_name not in state.restrictions:
            return True
        return len(state.restrictions[field_name]) > 0

    def _matches(self, actual: Any, expected: Any) -> bool:
        try:
            return str(actual).strip().lower() == str(expected).strip().lower()
        except:
            return actual == expected


if __name__ == "__main__":
    graph = ConstraintGraph([
        {"if": {"field": "status", "eq": "inactive"},
         "then": {"field": "verified", "must_be": False}},
        "if role == guest then access_level == read"
    ])
    state = graph.fresh_state()
    state = graph.commit_field("status", "inactive", state)
    assert graph.get_allowed_values("verified", None, state) == [False]
    state = graph.commit_field("role", "guest", state)
    assert graph.get_allowed_values("access_level", ["read", "write"], state) == ["read"]
    state2 = graph.fresh_state()
    state2 = graph.commit_field("status", "active", state2)
    assert graph.get_allowed_values("verified", None, state2) is None
    print("constraints.py: ALL TESTS PASSED")
