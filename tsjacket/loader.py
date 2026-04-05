import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass


@dataclass
class LoadedModel:
    model: any
    tokenizer: any
    vocab_size: int
    device: str


def load_model(model_name_or_path: str, device: str = "cpu") -> LoadedModel:
    print(f"Loading tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,
    )
    model.eval()
    model.to(device)

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        vocab_size=tokenizer.vocab_size,
        device=device
    )


def make_logits_fn(loaded: LoadedModel):
    def get_logits(input_ids: list[int]) -> torch.Tensor:
        with torch.no_grad():
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(loaded.device)
            outputs = loaded.model(input_tensor, use_cache=False)
            next_token_logits = outputs.logits[0, -1, :]
            return next_token_logits

    return get_logits


if __name__ == "__main__":
    print("Loading GPT-2 (this takes ~30 seconds first run)...")
    loaded = load_model("gpt2", device="cpu")

    print(f"Model loaded. Vocab size: {loaded.vocab_size}")

    test_ids = loaded.tokenizer.encode("Hello world", add_special_tokens=True)
    logits_fn = make_logits_fn(loaded)
    logits = logits_fn(test_ids)

    assert logits.shape == (loaded.vocab_size,), f"Expected ({loaded.vocab_size},) got {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits contain NaN"
    assert not torch.isinf(logits).any(), "Logits contain Inf"

    top_token_id = torch.argmax(logits).item()
    top_token_str = loaded.tokenizer.decode([top_token_id])
    print(f"Top next token after 'Hello world': {top_token_str!r}")

    print("loader: ALL TESTS PASSED")
