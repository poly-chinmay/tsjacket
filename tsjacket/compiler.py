from dataclasses import dataclass, field


@dataclass
class CompiledSchema:
    fields: list[str]
    required: set[str]
    field_types: dict[str, str]
    field_enums: dict[str, list[str] | None]


def compile_schema(schema: dict) -> CompiledSchema:
    if schema.get("type") != "object":
        raise ValueError("Only object schemas supported")

    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    for key, val in props.items():
        if "$ref" in val or "anyOf" in val or "oneOf" in val:
            raise NotImplementedError(f"Unsupported schema feature in field: {key}")

    fields = list(props.keys())
    field_types = {k: v["type"] for k, v in props.items()}
    field_enums = {k: v.get("enum") for k, v in props.items()}

    return CompiledSchema(
        fields=fields,
        required=required,
        field_types=field_types,
        field_enums=field_enums
    )


if __name__ == "__main__":
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "status": {"type": "string", "enum": ["active", "inactive"]}
        },
        "required": ["name", "age", "status"]
    }
    result = compile_schema(schema)
    print(result)
    assert result.fields == ["name", "age", "status"]
    assert result.required == {"name", "age", "status"}
    assert result.field_enums["status"] == ["active", "inactive"]
    assert result.field_enums["name"] is None
    print("compiler: ALL TESTS PASSED")
