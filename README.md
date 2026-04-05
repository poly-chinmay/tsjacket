# tsjacket

Constrained LLM decoding — invalid outputs don't get corrected, they never get produced.

## The Problem

LLMs are probabilistic functions being used as typed system components. When you ask a model to return structured JSON, it can produce malformed syntax, wrong types, or semantically invalid values. Most systems catch this after generation and retry. Retries don't converge when the failure is distributional. This is a systems problem, not a prompting problem.

## How It Works

Three layers run inside the generation loop:

```
Prompt → [Token Position]
		  ↓
	  Zone Classifier
	 /       |        \
 STRUCTURAL CONSTRAINED SEMANTIC
     ↓           ↓          ↓
  Force      Mask +      Free gen
  token     pressure    (type only)
     \           |          /
	└──── Next Token ────┘
		  ↓
	  Grammar Tracker
	  (advance state)
		  ↓
	  Checkpoint Save
	  (at field boundary)
```

**Zone Classifier** — at every token position, determines whether the valid token set is fully determined (STRUCTURAL), schema-bounded (CONSTRAINED), or free (SEMANTIC).

**Constraint Engine** — applies logit masking at CONSTRAINED positions. Records how much probability mass was masked — the pressure score.

**Pressure Monitor** — aggregates per-token pressure into per-field scores attached to every output. High pressure means the model was pushing toward invalid tokens. The value may be structurally correct but semantically suspect.

## What This Guarantees

- Structural validity: always
- Type correctness: always
- Per-field pressure scores: always attached to output
- Semantic correctness: not guaranteed — the model can still produce wrong values within valid structure

## Quickstart

```bash
pip install torch transformers
git clone https://github.com/poly-chinmay/tsjacket.git
cd tsjacket
pip install -e .
python examples/basic_usage.py
```

Or via CLI:

```bash
python cli.py --schema examples/example_schema.json \
		  --prompt "Generate a user profile" \
		  --show-pressure
```

## Example Output

```json
{
  "name": "Alice",
  "age": 31,
  "status": "active",
  "verified": true
}
```

```
Pressure Map:
  name         0.021
  age          0.743  ⚠  HIGH PRESSURE
  status       0.312
  verified     0.089
```

## What pressure_score means

At every CONSTRAINED token position, the engine records what fraction of the model's probability mass was sitting on invalid tokens before masking. High pressure on a field means the model was strongly pushing toward something the schema doesn't allow. The output is structurally valid regardless — but high pressure is a signal that the committed value is less reliable.

## Project Structure

| File | Responsibility |
|---|---|
| `tsjacket/compiler.py` | Converts JSON Schema to compiled constraint artifact |
| `tsjacket/bridge.py` | Maps grammar states to valid token IDs via prefix trie |
| `tsjacket/tracker.py` | Tracks grammar state as tokens are emitted |
| `tsjacket/zones.py` | Classifies each position as STRUCTURAL / CONSTRAINED / SEMANTIC |
| `tsjacket/engine.py` | Applies logit masking, records pressure score |
| `tsjacket/monitor.py` | Aggregates per-token pressure into per-field report |
| `tsjacket/checkpoints.py` | KV-cache checkpointing at field boundaries |
| `tsjacket/generator.py` | Orchestration loop — wires all modules together |
| `tsjacket/loader.py` | Model loading + logits function wrapper |
| `examples/basic_usage.py` | Runnable end-to-end example |
| `tests/test_harness.py` | Batch test suite with metrics |
| `cli.py` | Command-line interface |

## Limitations

- Requires logit-level model access — not compatible with GPT-4 or Claude APIs
- Recursive JSON Schema not supported
- Optional fields not yet supported
- Semantic validity is not enforced — only structural and type correctness

## Status

v0.1.0 — working constrained decoder with pressure monitoring.
Building in public. Follow along: [linkedin.com/in/chinmayaswale](https://linkedin.com/in/chinmayaswale)
