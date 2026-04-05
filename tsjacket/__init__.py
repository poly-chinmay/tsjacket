from tsjacket.compiler import compile_schema, CompiledSchema
from tsjacket.loader import load_model, make_logits_fn, LoadedModel
from tsjacket.bridge import build_token_trie
from tsjacket.generator import generate_constrained, GenerationResult


class ConstrainedGenerator:
    """
    Main entry point for Token Straitjacket.

    Usage:
        gen = ConstrainedGenerator(model="gpt2")
        result = gen.generate(
            prompt="Generate a user profile:",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "status": {"type": "string", "enum": ["active", "inactive"]}
                },
                "required": ["name", "age", "status"]
            }
        )
        print(result.json_output)
        print(result.pressure_map)
    """

    def __init__(self, model: str = "gpt2", device: str = "cpu"):
        self._loaded = load_model(model, device=device)
        self._logits_fn = make_logits_fn(self._loaded)
        self._trie_cache: dict = {}

    def generate(
        self,
        prompt: str,
        schema: dict,
        max_tokens: int = 150
    ) -> GenerationResult:
        compiled = compile_schema(schema)

        # Cache trie per schema (expensive to rebuild)
        schema_key = str(sorted(schema.get("properties", {}).keys()))
        if schema_key not in self._trie_cache:
            self._trie_cache[schema_key] = build_token_trie(
                compiled, self._loaded.tokenizer
            )
        trie = self._trie_cache[schema_key]

        prompt_ids = self._loaded.tokenizer.encode(
            prompt, add_special_tokens=True
        )

        return generate_constrained(
            self._logits_fn,
            prompt_ids,
            compiled,
            trie,
            self._loaded.tokenizer,
            max_new_tokens=max_tokens
        )


__version__ = "0.1.0"
__all__ = ["ConstrainedGenerator", "compile_schema", "GenerationResult"]
