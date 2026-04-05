import torch
from dataclasses import dataclass
from enum import Enum


class ZoneType(Enum):
    STRUCTURAL = "structural"
    CONSTRAINED = "constrained"
    SEMANTIC = "semantic"


class ConstraintDeadlockError(Exception):
    pass


@dataclass
class ConstraintResult:
    masked_logits: torch.Tensor
    pressure_score: float


def apply_constraint(
    logits: torch.Tensor,
    valid_token_ids: set[int] | None,
    zone_type: ZoneType,
    forced_token_id: int | None = None
) -> ConstraintResult:

    if zone_type == ZoneType.SEMANTIC:
        return ConstraintResult(masked_logits=logits, pressure_score=0.0)

    if zone_type == ZoneType.STRUCTURAL:
        assert forced_token_id is not None
        probs = torch.softmax(logits, dim=-1)
        valid_mass = probs[forced_token_id].item()
        masked = torch.full_like(logits, float('-inf'))
        masked[forced_token_id] = logits[forced_token_id]
        return ConstraintResult(masked_logits=masked, pressure_score=1.0 - valid_mass)

    if not valid_token_ids:
        raise ConstraintDeadlockError("No valid tokens for current grammar state")

    probs = torch.softmax(logits, dim=-1)
    valid_ids_list = list(valid_token_ids)
    valid_mass = probs[valid_ids_list].sum().item()
    pressure = 1.0 - valid_mass

    masked = torch.full_like(logits, float('-inf'))
    for tid in valid_token_ids:
        masked[tid] = logits[tid]

    return ConstraintResult(masked_logits=masked, pressure_score=round(pressure, 4))


if __name__ == "__main__":
    torch.manual_seed(42)
    vocab_size = 32000
    logits = torch.randn(vocab_size)
    valid = {42, 1337, 999, 17}

    result = apply_constraint(logits, valid, ZoneType.CONSTRAINED)

    invalid_ids = set(range(vocab_size)) - valid
    sample = list(invalid_ids)[:100]
    assert all(result.masked_logits[i].item() == float('-inf') for i in sample)
    assert 0.0 <= result.pressure_score <= 1.0

    result2 = apply_constraint(logits, None, ZoneType.STRUCTURAL, forced_token_id=42)
    valid_logits = [(i, result2.masked_logits[i].item()) for i in range(vocab_size) if result2.masked_logits[i].item() != float('-inf')]
    assert len(valid_logits) == 1
    assert valid_logits[0][0] == 42

    try:
        apply_constraint(logits, set(), ZoneType.CONSTRAINED)
        assert False, "Should have raised"
    except ConstraintDeadlockError:
        pass

    print("engine: ALL TESTS PASSED")
