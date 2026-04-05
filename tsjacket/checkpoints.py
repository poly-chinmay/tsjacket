import torch
import copy
from dataclasses import dataclass


@dataclass
class Checkpoint:
    field_index: int
    field_name: str
    token_index: int
    input_ids_snapshot: list[int]
    grammar_state_snapshot: str
    committed_fields_snapshot: dict


class CheckpointManager:
    def __init__(self, max_checkpoints: int = 10):
        self._stack: list[Checkpoint] = []
        self._max = max_checkpoints

    def save(
        self,
        field_index: int,
        field_name: str,
        token_index: int,
        input_ids: list[int],
        grammar_state: str,
        committed_fields: dict
    ):
        if len(self._stack) >= self._max:
            self._stack.pop(0)

        self._stack.append(Checkpoint(
            field_index=field_index,
            field_name=field_name,
            token_index=token_index,
            input_ids_snapshot=list(input_ids),
            grammar_state_snapshot=grammar_state,
            committed_fields_snapshot=dict(committed_fields)
        ))

    def rollback(self) -> Checkpoint | None:
        if not self._stack:
            return None
        return self._stack.pop()

    def peek(self) -> Checkpoint | None:
        if not self._stack:
            return None
        return self._stack[-1]

    def depth(self) -> int:
        return len(self._stack)

    def clear(self):
        self._stack = []


if __name__ == "__main__":
    manager = CheckpointManager(max_checkpoints=5)
    fields = ["name", "age", "status", "verified"]
    for i, fname in enumerate(fields):
        manager.save(
            field_index=i,
            field_name=fname,
            token_index=i * 5,
            input_ids=[1, 2, 3, i],
            grammar_state="expect_key",
            committed_fields={f: "val" for f in fields[:i]}
        )

    assert manager.depth() == 4

    cp = manager.rollback()
    assert cp.field_name == "verified"
    assert manager.depth() == 3

    cp2 = manager.rollback()
    assert cp2.field_name == "status"
    assert cp2.field_index == 2
    assert cp2.input_ids_snapshot == [1, 2, 3, 2]

    original_ids = cp2.input_ids_snapshot
    original_ids.append(999)
    cp3 = manager.rollback()
    assert 999 not in cp3.input_ids_snapshot

    manager2 = CheckpointManager(max_checkpoints=2)
    for i in range(5):
        manager2.save(i, f"field_{i}", i, [i], "state", {})
    assert manager2.depth() == 2
    cp = manager2.rollback()
    assert cp.field_name == "field_4"

    manager3 = CheckpointManager()
    assert manager3.rollback() is None

    print("checkpoints: ALL TESTS PASSED")
