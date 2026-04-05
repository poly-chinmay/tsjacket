"""
Token Straitjacket — Basic Usage Example
github.com/poly-chinmay/tsjacket
"""

from tsjacket import ConstrainedGenerator
import json

# Load model once — reuse for multiple generations
gen = ConstrainedGenerator(model="gpt2", device="cpu")

# Define your schema
schema = {
    "type": "object",
    "properties": {
        "name":     {"type": "string"},
        "age":      {"type": "integer"},
        "status":   {"type": "string", "enum": ["active", "inactive"]},
        "verified": {"type": "boolean"}
    },
    "required": ["name", "age", "status", "verified"]
}

# Generate — always returns valid JSON
result = gen.generate(
    prompt="Generate a JSON user profile:",
    schema=schema
)

print("Output JSON:")
print(json.dumps(result.json_output, indent=2))

print("\nPressure Map (how hard the model fought the constraint):")
for field, data in result.pressure_map.items():
    flag = " ⚠  HIGH PRESSURE" if data["flagged"] else ""
    print(f"  {field:<12} {data['mean_pressure']:.3f}{flag}")

print(f"\nParse success:  {result.parse_success}")
print(f"Rollback count: {result.rollback_count}")
