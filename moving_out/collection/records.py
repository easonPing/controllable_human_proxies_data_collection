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
STUDY_SCHEMA_VERSION = 4
CONSENT_SCHEMA_VERSION = 2


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


@dataclass
class ConsentRecord:
    """Electronic consent acknowledgement stored separately from trajectories."""

    consent_id: str
    computing_id: str
    email: str
    agreed_at: str
    document_name: str
    document_sha256: str
    schema_version: int = CONSENT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "consent_id": self.consent_id,
            "computing_id": self.computing_id,
            "email": self.email,
            "agreed_at": self.agreed_at,
            "document_name": self.document_name,
            "document_sha256": self.document_sha256,
        }


@dataclass
class StudyEpisode:
    """One map/condition pair and its independently indexed trajectory."""

    metadata: Dict[str, Any]
    trajectory: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = dict(to_jsonable(self.metadata))
        payload["trajectory"] = to_jsonable(self.trajectory)
        return payload


@dataclass
class StudyQuestionnaireItem:
    """One response in a condition-specific study questionnaire."""

    question: str
    options: List[str]
    answer: Optional[str] = None
    skipped: bool = False

    def __post_init__(self) -> None:
        if not self.question:
            raise ValueError("Question text must not be empty.")
        if not self.options:
            raise ValueError("A question must contain at least one option.")
        if self.answer is not None and self.answer not in self.options:
            raise ValueError("The answer must be one of the configured options.")
        if self.answer is not None and self.skipped:
            raise ValueError("A questionnaire item cannot be answered and skipped.")


@dataclass
class StudyQuestionnaire:
    """A complete copy of the questionnaire for one action condition."""

    metadata: Dict[str, Any]
    items: List[StudyQuestionnaireItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = dict(to_jsonable(self.metadata))
        payload["items"] = [
            to_jsonable(dataclasses.asdict(item)) for item in self.items
        ]
        return payload


@dataclass
class StudyRecord:
    """Multi-episode persisted representation of the formal study workflow."""

    metadata: Dict[str, Any]
    tutorial: Dict[str, Any] = field(default_factory=dict)
    episodes: List[StudyEpisode] = field(default_factory=list)
    questionnaires: List[StudyQuestionnaire] = field(default_factory=list)
    schema_version: int = STUDY_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metadata": to_jsonable(self.metadata),
            "tutorial": to_jsonable(self.tutorial),
            "episodes": [episode.to_dict() for episode in self.episodes],
            "questionnaires": [
                questionnaire.to_dict()
                for questionnaire in self.questionnaires
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


def create_study_path(
    output_dir: str,
    session_id: str,
    timestamp: Optional[dt.datetime] = None,
) -> str:
    """Build a collision-resistant multi-episode study filename."""

    timestamp = timestamp or dt.datetime.now().astimezone()
    file_name = (
        f"study_{timestamp.strftime('%Y%m%d%H%M%S')}_{session_id[:8]}.json"
    )
    return os.path.abspath(os.path.join(os.path.expanduser(output_dir), file_name))


def create_consent_path(output_dir: str, consent_id: str) -> str:
    """Build a non-PII filename in the dedicated consent directory."""

    return os.path.abspath(
        os.path.join(
            os.path.expanduser(output_dir),
            "consents",
            f"consent_{consent_id}.json",
        )
    )


def save_session(path: str, record: SessionRecord) -> None:
    """Atomically write a session so readers never observe partial JSON."""

    _save_json_payload(path, record.to_dict())


def save_study(path: str, record: StudyRecord) -> None:
    """Atomically write a schema-v4 study record."""

    _save_json_payload(path, record.to_dict())


def save_consent(path: str, record: ConsentRecord) -> None:
    """Atomically write a separately stored electronic consent record."""

    _save_json_payload(path, record.to_dict())


def _save_json_payload(path: str, payload: Dict[str, Any]) -> None:
    """Atomically write one JSON payload."""

    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    temporary_path = os.path.join(
        output_dir,
        f".{os.path.basename(path)}.{uuid.uuid4().hex}.tmp",
    )
    try:
        with open(temporary_path, "x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
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
