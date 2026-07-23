"""Versioned session records for human-AI data collection."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 3


@dataclass(frozen=True)
class CollectionConfig:
    """Configuration selected before a collection session begins."""

    computing_id: str
    email: str
    human_agent: str
    ai_model: str
    map_id: str
    input_mode: str = "joystick"

    @property
    def human_agent_index(self) -> int:
        if self.human_agent == "blue":
            return 0
        if self.human_agent == "red":
            return 1
        raise ValueError(f"Unknown human agent: {self.human_agent!r}")

    @property
    def ai_agent(self) -> str:
        if self.human_agent == "blue":
            return "red"
        if self.human_agent == "red":
            return "blue"
        raise ValueError(f"Unknown human agent: {self.human_agent!r}")

    @property
    def ai_agent_index(self) -> int:
        return 1 - self.human_agent_index


@dataclass
class QuestionnaireItem:
    """One single-choice questionnaire item and its optional response."""

    question: str
    options: List[str]
    answer: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.question:
            raise ValueError("Question text must not be empty.")
        if not self.options:
            raise ValueError("A question must contain at least one option.")
        if len(set(self.options)) != len(self.options):
            raise ValueError("Question options must be unique.")
        if self.answer is not None and self.answer not in self.options:
            raise ValueError("The answer must be one of the configured options.")


@dataclass
class SessionRecord:
    """Complete persisted representation of one collection session."""

    metadata: Dict[str, Any]
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    questionnaire: List[QuestionnaireItem] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metadata": to_jsonable(self.metadata),
            "trajectory": to_jsonable(self.trajectory),
            "questionnaire": [
                to_jsonable(dataclasses.asdict(item)) for item in self.questionnaire
            ],
        }


def now_iso() -> str:
    """Return a timezone-aware ISO-8601 timestamp."""

    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def new_session_id() -> str:
    return str(uuid.uuid4())


def create_session_path(
    output_dir: str,
    map_id: str,
    session_id: str,
    timestamp: Optional[dt.datetime] = None,
) -> str:
    """Build a collision-resistant filename without embedding participant PII."""

    timestamp = timestamp or dt.datetime.now().astimezone()
    safe_map = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in map_id)
    file_name = (
        f"{safe_map}_{timestamp.strftime('%Y%m%d%H%M%S')}_{session_id[:8]}.json"
    )
    return os.path.abspath(os.path.join(os.path.expanduser(output_dir), file_name))


def save_session(path: str, record: SessionRecord) -> None:
    """Atomically write a session so readers never observe partial JSON."""

    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    temporary_path = os.path.join(
        output_dir,
        f".{os.path.basename(path)}.{uuid.uuid4().hex}.tmp",
    )
    try:
        with open(temporary_path, "x", encoding="utf-8") as stream:
            json.dump(record.to_dict(), stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def to_jsonable(value: Any) -> Any:
    """Recursively convert NumPy/Pymunk-style values to JSON primitives."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return to_jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")
