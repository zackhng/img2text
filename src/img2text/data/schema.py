"""Unified JSONL record shared by every dataset converter and the trainer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Literal

Task = Literal["table", "chart", "diagram", "other"]
TargetFormat = Literal["otsl", "html", "chartcsv", "mermaid", "text"]

TASKS: tuple[str, ...] = ("table", "chart", "diagram", "other")
TARGET_FORMATS: tuple[str, ...] = ("otsl", "html", "chartcsv", "mermaid", "text")


@dataclass
class Record:
    id: str
    image: str  # path to the image crop, relative to the dataset root
    task: Task
    target: str
    target_format: TargetFormat
    source: str
    split: str = "train"
    meta: dict = field(default_factory=dict)  # generator settings, free-form

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}")
        if self.target_format not in TARGET_FORMATS:
            raise ValueError(f"unknown target_format {self.target_format!r}")


def write_jsonl(records: Iterable[Record], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Record]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Record(**json.loads(line))
