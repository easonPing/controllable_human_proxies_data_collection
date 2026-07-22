"""Socket protocol, model registry, and child-process AI runtime."""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import random
import socket
import struct
import time
from abc import ABC, abstractmethod
from numbers import Real
from typing import Any, Dict, List, Optional, Sequence, Type

from .records import to_jsonable


PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 2 * 1024 * 1024
HEADER = struct.Struct("!I")


class AIModel(ABC):
    """Interface implemented by every AI model exposed in the UI."""

    @abstractmethod
    def reset(self, agent_index: int, seed: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, state: Dict[str, Any]) -> Sequence[Any]:
        raise NotImplementedError


class RandomModel(AIModel):
    """Placeholder model that emits seeded, valid random controls."""

    def __init__(self) -> None:
        self._rng = random.Random()
        self.agent_index = 0

    def reset(self, agent_index: int, seed: int) -> None:
        self.agent_index = agent_index
        self._rng.seed(seed)

    def predict(self, state: Dict[str, Any]) -> Sequence[Any]:
        del state
        return [
            self._rng.uniform(0.0, 1.0),
            self._rng.uniform(-math.pi, math.pi),
            self._rng.random() < 0.05,
        ]


MODEL_REGISTRY: Dict[str, Type[AIModel]] = {"random": RandomModel}


def validate_action(action: Sequence[Any]) -> List[Any]:
    """Validate and normalize an action received across the process boundary."""

    if not isinstance(action, (list, tuple)) or len(action) != 3:
        raise ValueError("AI action must be a three-item list.")
    magnitude, angle, interact = action
    if (
        isinstance(magnitude, bool)
        or not isinstance(magnitude, Real)
        or not math.isfinite(float(magnitude))
        or not -1.0 <= float(magnitude) <= 1.0
    ):
        raise ValueError("AI action magnitude must be finite and in [-1, 1].")
    if (
        isinstance(angle, bool)
        or not isinstance(angle, Real)
        or not math.isfinite(float(angle))
        or not -math.pi <= float(angle) <= math.pi
    ):
        raise ValueError("AI action angle must be finite and in [-pi, pi].")
    if not isinstance(interact, bool):
        raise ValueError("AI interact action must be boolean.")
    return [float(magnitude), float(angle), interact]


def encode_message(message: Dict[str, Any]) -> bytes:
    payload = json.dumps(
        to_jsonable(message), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"Socket message exceeds {MAX_MESSAGE_BYTES} bytes.")
    return HEADER.pack(len(payload)) + payload


def send_message(sock: socket.socket, message: Dict[str, Any]) -> None:
    sock.sendall(encode_message(message))


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = sock.recv(length - len(chunks))
        if not chunk:
            raise ConnectionError("AI socket closed while receiving a message.")
        chunks.extend(chunk)
    return bytes(chunks)


def receive_message(sock: socket.socket) -> Dict[str, Any]:
    payload_length = HEADER.unpack(_recv_exact(sock, HEADER.size))[0]
    if payload_length > MAX_MESSAGE_BYTES:
        raise ValueError(f"Incoming socket message exceeds {MAX_MESSAGE_BYTES} bytes.")
    payload = _recv_exact(sock, payload_length)
    message = json.loads(payload.decode("utf-8"))
    if not isinstance(message, dict):
        raise ValueError("Socket message must decode to an object.")
    return message


class JSONMessageBuffer:
    """Incrementally parse length-prefixed JSON from a non-blocking socket."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._expected_length: Optional[int] = None

    def feed(self, data: bytes) -> List[Dict[str, Any]]:
        self._buffer.extend(data)
        messages: List[Dict[str, Any]] = []
        while True:
            if self._expected_length is None:
                if len(self._buffer) < HEADER.size:
                    break
                self._expected_length = HEADER.unpack(self._buffer[: HEADER.size])[0]
                del self._buffer[: HEADER.size]
                if self._expected_length > MAX_MESSAGE_BYTES:
                    raise ValueError(
                        f"Incoming socket message exceeds {MAX_MESSAGE_BYTES} bytes."
                    )
            if len(self._buffer) < self._expected_length:
                break
            payload = bytes(self._buffer[: self._expected_length])
            del self._buffer[: self._expected_length]
            self._expected_length = None
            message = json.loads(payload.decode("utf-8"))
            if not isinstance(message, dict):
                raise ValueError("Socket message must decode to an object.")
            messages.append(message)
        return messages

    def read_available(self, sock: socket.socket) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        while True:
            try:
                chunk = sock.recv(65536)
            except (BlockingIOError, InterruptedError):
                break
            if not chunk:
                raise ConnectionError("AI socket closed.")
            messages.extend(self.feed(chunk))
            if len(chunk) < 65536:
                break
        return messages


def run_ai_worker(
    host: str,
    port: int,
    model_name: str,
    agent_index: int,
    seed: int,
) -> None:
    """Child-process entry point. It intentionally has no UI/environment imports."""

    model_class = MODEL_REGISTRY.get(model_name)
    if model_class is None:
        return
    model = model_class()
    model.reset(agent_index=agent_index, seed=seed)

    deadline = time.monotonic() + 5.0
    connection: Optional[socket.socket] = None
    while connection is None and time.monotonic() < deadline:
        try:
            connection = socket.create_connection((host, port), timeout=1.0)
        except OSError:
            time.sleep(0.05)
    if connection is None:
        return

    with connection:
        connection.settimeout(None)
        send_message(
            connection,
            {
                "type": "ready",
                "protocol_version": PROTOCOL_VERSION,
                "model": model_name,
            },
        )
        while True:
            try:
                message = receive_message(connection)
                message_type = message.get("type")
                if message_type == "shutdown":
                    send_message(connection, {"type": "shutdown_ack"})
                    return
                if message_type != "act":
                    raise ValueError(f"Unsupported message type: {message_type!r}")
                request_id = message.get("request_id")
                state = message.get("state")
                if not isinstance(request_id, int) or not isinstance(state, dict):
                    raise ValueError("Invalid act request.")
                action = validate_action(model.predict(state))
                send_message(
                    connection,
                    {
                        "type": "action",
                        "request_id": request_id,
                        "action": action,
                    },
                )
            except Exception as exc:
                try:
                    send_message(
                        connection,
                        {"type": "error", "message": str(exc)},
                    )
                except OSError:
                    pass
                return


class AIProcessClient:
    """Own an AI worker process and its non-blocking request/response channel."""

    def __init__(self, response_timeout: float = 2.0) -> None:
        if response_timeout <= 0:
            raise ValueError("AI response timeout must be positive.")
        self.response_timeout = response_timeout
        self._listener: Optional[socket.socket] = None
        self._connection: Optional[socket.socket] = None
        self._process: Optional[mp.Process] = None
        self._reader = JSONMessageBuffer()
        self._pending_request_id: Optional[int] = None
        self._request_started_at: Optional[float] = None

    @property
    def pending(self) -> bool:
        return self._pending_request_id is not None

    def start(
        self,
        model_name: str,
        agent_index: int,
        seed: int,
        startup_timeout: float = 5.0,
    ) -> None:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown AI model: {model_name!r}")
        if self._process is not None:
            raise RuntimeError("AI process is already started.")

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(startup_timeout)
        host, port = listener.getsockname()
        self._listener = listener

        context = mp.get_context("spawn")
        process = context.Process(
            target=run_ai_worker,
            args=(host, port, model_name, agent_index, seed),
            name="moving-out-ai",
        )
        process.start()
        self._process = process

        try:
            connection, _ = listener.accept()
            connection.settimeout(startup_timeout)
            ready = receive_message(connection)
            if (
                ready.get("type") != "ready"
                or ready.get("protocol_version") != PROTOCOL_VERSION
                or ready.get("model") != model_name
            ):
                raise RuntimeError(f"Invalid AI ready message: {ready!r}")
            connection.setblocking(False)
            self._connection = connection
        except Exception:
            self.close()
            raise
        finally:
            listener.close()
            self._listener = None

    def request_action(self, request_id: int, state: Dict[str, Any]) -> None:
        if self._connection is None:
            raise RuntimeError("AI process is not connected.")
        if self.pending:
            raise RuntimeError(
                "Cannot send an action request while another is pending."
            )
        self._connection.settimeout(0.5)
        try:
            send_message(
                self._connection,
                {"type": "act", "request_id": request_id, "state": state},
            )
        finally:
            self._connection.setblocking(False)
        self._pending_request_id = request_id
        self._request_started_at = time.monotonic()

    def poll_action(self) -> Optional[List[Any]]:
        if not self.pending:
            return None
        if self._connection is None:
            raise ConnectionError("AI process is not connected.")
        messages = self._reader.read_available(self._connection)
        for message in messages:
            if message.get("type") == "error":
                raise RuntimeError(
                    f"AI worker error: {message.get('message', 'unknown')}"
                )
            if message.get("type") != "action":
                raise RuntimeError(f"Unexpected AI message: {message!r}")
            if message.get("request_id") != self._pending_request_id:
                raise RuntimeError("AI response request_id does not match the request.")
            action = validate_action(message.get("action"))
            self._pending_request_id = None
            self._request_started_at = None
            return action
        if (
            self._request_started_at is not None
            and time.monotonic() - self._request_started_at > self.response_timeout
        ):
            raise TimeoutError(
                f"AI response exceeded {self.response_timeout:.1f} seconds."
            )
        return None

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.settimeout(0.25)
                send_message(connection, {"type": "shutdown"})
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass

        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None

        process = self._process
        self._process = None
        if process is not None:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

        self._pending_request_id = None
        self._request_started_at = None
