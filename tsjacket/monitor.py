from dataclasses import dataclass, field


@dataclass
class TokenPressureRecord:
    token_index: int
    grammar_state: str
    field_name: str | None
    pressure_score: float


@dataclass
class FieldPressureReport:
    field_name: str
    mean_pressure: float
    max_pressure: float
    token_count: int
    flagged: bool


class PressureMonitor:
    def __init__(self):
        self._records: list[TokenPressureRecord] = []

    def record(self, token_index: int, grammar_state: str,
               field_name: str | None, pressure_score: float):
        self._records.append(TokenPressureRecord(
            token_index=token_index,
            grammar_state=grammar_state,
            field_name=field_name,
            pressure_score=round(pressure_score, 4)
        ))

    def build_report(self) -> dict[str, FieldPressureReport]:
        grouped: dict[str, list[float]] = {}
        for r in self._records:
            if r.field_name is None:
                continue
            if r.field_name not in grouped:
                grouped[r.field_name] = []
            grouped[r.field_name].append(r.pressure_score)

        report = {}
        for fname, scores in grouped.items():
            mean_p = sum(scores) / len(scores)
            report[fname] = FieldPressureReport(
                field_name=fname,
                mean_pressure=round(mean_p, 4),
                max_pressure=round(max(scores), 4),
                token_count=len(scores),
                flagged=mean_p > 0.6
            )
        return report

    def format_output(self) -> dict:
        report = self.build_report()
        return {
            fname: {
                "mean_pressure": r.mean_pressure,
                "max_pressure": r.max_pressure,
                "token_count": r.token_count,
                "flagged": r.flagged
            }
            for fname, r in report.items()
        }

    def reset(self):
        self._records = []


if __name__ == "__main__":
    monitor = PressureMonitor()
    monitor.record(0, "start", None, 0.0)
    monitor.record(1, "expect_value:age", "age", 0.92)
    monitor.record(2, "expect_value:age", "age", 0.88)
    monitor.record(3, "expect_value:age", "age", 0.79)
    monitor.record(4, "expect_value:name", "name", 0.05)
    monitor.record(5, "expect_value:name", "name", 0.12)
    monitor.record(6, "expect_value:status", "status", 0.55)

    report = monitor.build_report()

    assert report["age"].flagged is True
    assert report["name"].flagged is False
    assert report["age"].mean_pressure == round((0.92 + 0.88 + 0.79) / 3, 4)
    assert report["age"].token_count == 3
    assert report["status"].flagged is False

    output = monitor.format_output()
    print("Pressure output:")
    import json
    print(json.dumps(output, indent=2))

    print("monitor: ALL TESTS PASSED")
