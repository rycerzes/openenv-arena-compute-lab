# SPDX-License-Identifier: BSD-3-Clause

"""Task instance types and the answer grader."""

import csv
import datetime
import io
import math
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Callable

_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass
class AnswerField:
    """One graded part of the answer."""

    name: str
    expected: Any
    kind: str  # "number", "int" or "text"
    description: str
    rel_tol: float = 0.0
    abs_tol: float = 0.0
    choices: list[str] = field(default_factory=list)


@dataclass
class TaskInstance:
    """A generated task: what the agent sees and what the grader expects."""

    question: str
    files: dict[str, str]
    fields: list[AnswerField]


@dataclass(frozen=True)
class Family:
    """A task family with three difficulty levels."""

    name: str
    domain: str
    summary: str
    generate: Callable[[int, Any], TaskInstance]


def cents(x: float) -> float:
    """Round to cents, half up."""
    return float(Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _to_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        match = _NUMBER.search(value.replace(",", "").replace("_", ""))
        if match:
            return float(match.group())
    return None


def field_correct(spec: AnswerField, value: Any) -> bool:
    if spec.kind == "text":
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            return False
        got = str(value).strip()
        if got == str(spec.expected):
            return True
        folded = [c for c in spec.choices if c.lower() == got.lower()]
        return len(folded) == 1 and folded[0] == spec.expected
    got = _to_number(value)
    if got is None:
        return False
    if spec.kind == "int":
        return got == float(spec.expected)
    tol = max(spec.abs_tol, spec.rel_tol * abs(float(spec.expected)))
    return abs(got - float(spec.expected)) <= tol


def grade(fields: list[AnswerField], answer: dict[str, Any]) -> tuple[float, list[str]]:
    """Return the share of correct fields and the names of the wrong ones."""
    wrong = [f.name for f in fields if not field_correct(f, answer.get(f.name))]
    return (len(fields) - len(wrong)) / len(fields), wrong


def csv_text(header: list[str], rows: list[list[Any]]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return out.getvalue()


def accounting(amount: float, decimals: int = 2, symbol: str = "") -> str:
    """1234.5 -> '1,234.50'; negatives in parentheses: '(1,234.50)'."""
    text = f"{symbol}{abs(amount):,.{decimals}f}"
    return f"({text})" if amount < 0 else text


def day_first(day: datetime.date) -> str:
    return day.strftime("%d/%m/%Y")
