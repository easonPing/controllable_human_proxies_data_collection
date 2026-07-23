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
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type

from .records import to_jsonable


PROTOCOL_VERSION = 2
MAX_MESSAGE_BYTES = 2 * 1024 * 1024
HEADER = struct.Struct("!I")


@dataclass(frozen=True)
class ActionPlan:
    """A model prediction plus its preferred re-planning chunk size."""

    actions: List[List[Any]]
    preferred_action_count: int


@dataclass(frozen=True)
class PlanResult:
    """A validated plan returned across the worker-process boundary."""

    request_id: int
    source_step: int
    actions: List[List[Any]]
    preferred_action_count: int


class AIModel(ABC):
    """Interface implemented by every AI model exposed in the UI."""

    startup_timeout = 5.0
    minimum_response_timeout = 0.0
    plan_horizon = 1
    preferred_action_count = 1

    @abstractmethod
    def reset(self, agent_index: int, seed: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_plan(self, state: Dict[str, Any]) -> ActionPlan:
        raise NotImplementedError

    def predict(self, state: Dict[str, Any]) -> Sequence[Any]:
        """Compatibility helper returning the first action of a plan."""
        plan = self.predict_plan(state)
        if not plan.actions:
            raise ValueError("AI model returned an empty action plan.")
        return plan.actions[0]


class RandomModel(AIModel):
    """Placeholder model that emits seeded, valid random controls."""

    def __init__(self) -> None:
        self._rng = random.Random()
        self.agent_index = 0

    def reset(self, agent_index: int, seed: int) -> None:
        self.agent_index = agent_index
        self._rng.seed(seed)

    def predict_plan(self, state: Dict[str, Any]) -> ActionPlan:
        del state
        return ActionPlan(
            actions=[
                [
                    self._rng.uniform(0.0, 1.0),
                    self._rng.uniform(-math.pi, math.pi),
                    self._rng.random() < 0.05,
                ]
            ],
            preferred_action_count=1,
        )


class DiffusionPolicyHandOffModel(AIModel):
    """Adapter for the two shipped HandOff diffusion-policy checkpoints."""

    startup_timeout = 30.0
    minimum_response_timeout = 10.0
    observation_dim = 121
    plan_horizon = 8
    preferred_action_count = 2
    num_inference_steps = 32
    hold_scale = 5.0

    def __init__(self) -> None:
        self.agent_index: Optional[int] = None
        self._encoder: Optional[Any] = None
        self._policy: Optional[Any] = None

    @staticmethod
    def checkpoint_path(agent_index: int) -> Path:
        if agent_index not in (0, 1):
            raise ValueError("Diffusion policy agent_index must be 0 or 1.")
        repository_root = Path(__file__).resolve().parents[2]
        return (
            repository_root
            / "movingout_dp_deploy"
            / f"dp_handoff_robot{agent_index + 1}.pt"
        )

    @staticmethod
    def _seed_torch(torch_module: Any, seed: int) -> None:
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)

    def reset(self, agent_index: int, seed: int) -> None:
        checkpoint_path = self.checkpoint_path(agent_index)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Diffusion checkpoint not found: {checkpoint_path}"
            )

        try:
            import torch

            from moving_out.utils.states_encoding import StatesEncoder
            from movingout_dp_deploy import DiffusionPolicy
        except ImportError as exc:
            raise RuntimeError(
                "Could not import diffusion-policy dependencies. "
                "Install requirements.txt in the active environment."
            ) from exc

        self.agent_index = agent_index
        self._encoder = StatesEncoder(max_number_shaps=7)
        self._policy = DiffusionPolicy(
            str(checkpoint_path),
            device=None,
            hold_scale=self.hold_scale,
            num_inference_steps=self.num_inference_steps,
        )

        # Pay the one-time Torch/kernel initialization cost before reporting
        # the worker as ready, then restore both policy and RNG state.
        self._seed_torch(torch, seed)
        self._policy.get_action_plan([0.0] * self.observation_dim)
        self._policy.reset()
        self._seed_torch(torch, seed)

    def predict_plan(self, state: Dict[str, Any]) -> ActionPlan:
        if (
            self.agent_index is None
            or self._encoder is None
            or self._policy is None
        ):
            raise RuntimeError("Diffusion policy must be reset before prediction.")
        encoded_states = self._encoder.get_state_by_current_obs_states(state)
        observation = encoded_states[self.agent_index]
        if len(observation) != self.observation_dim:
            raise ValueError(
                "Diffusion policy expected a "
                f"{self.observation_dim}-dimensional observation, "
                f"but received {len(observation)} values."
            )
        actions = self._policy.get_action_plan(observation)
        if len(actions) != self.plan_horizon:
            raise ValueError(
                "Diffusion policy expected to return "
                f"{self.plan_horizon} actions, but returned {len(actions)}."
            )
        return ActionPlan(
            actions=[list(action) for action in actions],
            preferred_action_count=min(
                self.preferred_action_count, self.plan_horizon
            ),
        )


MODEL_REGISTRY: Dict[str, Type[AIModel]] = {
    "random": RandomModel,
    "diffusion_policy_hand_off": DiffusionPolicyHandOffModel,
}


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


def validate_action_plan(plan: ActionPlan) -> ActionPlan:
    """Validate every action and the preferred chunk size in a model plan."""

    if not isinstance(plan, ActionPlan):
        raise ValueError("AI model must return an ActionPlan.")
    if not isinstance(plan.actions, (list, tuple)) or not plan.actions:
        raise ValueError("AI action plan must contain at least one action.")
    if (
        isinstance(plan.preferred_action_count, bool)
        or not isinstance(plan.preferred_action_count, int)
        or not 1 <= plan.preferred_action_count <= len(plan.actions)
    ):
        raise ValueError(
            "AI preferred_action_count must be between 1 and the plan length."
        )
    return ActionPlan(
        actions=[validate_action(action) for action in plan.actions],
        preferred_action_count=plan.preferred_action_count,
    )


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
        try:
            if model_class is None:
                raise ValueError(f"Unknown AI model: {model_name!r}")
            model = model_class()
            model.reset(agent_index=agent_index, seed=seed)
        except Exception as exc:
            try:
                send_message(
                    connection,
                    {
                        "type": "startup_error",
                        "protocol_version": PROTOCOL_VERSION,
                        "model": model_name,
                        "message": str(exc),
                    },
                )
            except OSError:
                pass
            return

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
                if message_type != "plan":
                    raise ValueError(f"Unsupported message type: {message_type!r}")
                request_id = message.get("request_id")
                source_step = message.get("source_step")
                state = message.get("state")
                if (
                    isinstance(request_id, bool)
                    or not isinstance(request_id, int)
                    or isinstance(source_step, bool)
                    or not isinstance(source_step, int)
                    or source_step < 0
                    or not isinstance(state, dict)
                ):
                    raise ValueError("Invalid plan request.")
                plan = validate_action_plan(model.predict_plan(state))
                send_message(
                    connection,
                    {
                        "type": "plan_result",
                        "request_id": request_id,
                        "source_step": source_step,
                        "actions": plan.actions,
                        "preferred_action_count": plan.preferred_action_count,
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
        self._pending_source_step: Optional[int] = None
        self._request_started_at: Optional[float] = None
        self._outgoing = bytearray()

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
            self._connection = connection
            connection.settimeout(startup_timeout)
            ready = receive_message(connection)
            if ready.get("type") == "startup_error":
                raise RuntimeError(
                    "AI worker could not start "
                    f"{model_name!r}: {ready.get('message', 'unknown error')}"
                )
            if (
                ready.get("type") != "ready"
                or ready.get("protocol_version") != PROTOCOL_VERSION
                or ready.get("model") != model_name
            ):
                raise RuntimeError(f"Invalid AI ready message: {ready!r}")
            connection.setblocking(False)
        except Exception:
            self.close()
            raise
        finally:
            listener.close()
            self._listener = None

    def _flush_outgoing(self) -> None:
        if not self._outgoing:
            return
        if self._connection is None:
            raise ConnectionError("AI process is not connected.")
        while self._outgoing:
            try:
                sent = self._connection.send(self._outgoing)
            except (BlockingIOError, InterruptedError):
                return
            if sent <= 0:
                raise ConnectionError("AI socket closed while sending a request.")
            del self._outgoing[:sent]

    def request_plan(
        self,
        request_id: int,
        source_step: int,
        state: Dict[str, Any],
    ) -> None:
        if self._connection is None:
            raise RuntimeError("AI process is not connected.")
        if self.pending:
            raise RuntimeError(
                "Cannot send a plan request while another is pending."
            )
        if (
            isinstance(request_id, bool)
            or not isinstance(request_id, int)
            or isinstance(source_step, bool)
            or not isinstance(source_step, int)
            or source_step < 0
            or not isinstance(state, dict)
        ):
            raise ValueError("Invalid plan request.")
        self._outgoing.extend(
            encode_message(
                {
                    "type": "plan",
                    "request_id": request_id,
                    "source_step": source_step,
                    "state": state,
                }
            )
        )
        self._pending_request_id = request_id
        self._pending_source_step = source_step
        self._request_started_at = time.monotonic()
        self._flush_outgoing()

    def poll_plan(self) -> Optional[PlanResult]:
        if not self.pending:
            return None
        if self._connection is None:
            raise ConnectionError("AI process is not connected.")
        self._flush_outgoing()
        messages = self._reader.read_available(self._connection)
        for message in messages:
            if message.get("type") == "error":
                raise RuntimeError(
                    f"AI worker error: {message.get('message', 'unknown')}"
                )
            if message.get("type") != "plan_result":
                raise RuntimeError(f"Unexpected AI message: {message!r}")
            if message.get("request_id") != self._pending_request_id:
                raise RuntimeError("AI response request_id does not match the request.")
            if message.get("source_step") != self._pending_source_step:
                raise RuntimeError(
                    "AI response source_step does not match the request."
                )
            actions = message.get("actions")
            preferred_action_count = message.get("preferred_action_count")
            plan = validate_action_plan(
                ActionPlan(
                    actions=actions,
                    preferred_action_count=preferred_action_count,
                )
            )
            result = PlanResult(
                request_id=self._pending_request_id,
                source_step=self._pending_source_step,
                actions=plan.actions,
                preferred_action_count=plan.preferred_action_count,
            )
            self._pending_request_id = None
            self._pending_source_step = None
            self._request_started_at = None
            return result
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
        self._pending_source_step = None
        self._request_started_at = None
        self._outgoing.clear()
