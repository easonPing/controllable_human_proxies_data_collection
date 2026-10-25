"""Multi-stage consent, tutorial, and formal Moving Out study workflow."""

from __future__ import annotations

import hashlib
import math
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pygame

from moving_out.benchmarks.moving_out import MovingOutEnv
from moving_out.utils.joystick_controller import JoystickController
from moving_out.utils.keyboard_controller import KeyboardController

from .ai_runtime import AIProcessClient, MODEL_REGISTRY
from .app import (
    CONTROL_INTERVAL,
    DEFAULT_QUESTIONS,
    IDLE_AI_ACTION,
    UI_FPS,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    Button,
    CollectionApp,
    compose_joint_action,
    validate_email,
)
from .records import (
    CollectionConfig,
    ConsentRecord,
    QuestionnaireItem,
    SessionRecord,
    StudyEpisode,
    StudyQuestionnaire,
    StudyQuestionnaireItem,
    StudyRecord,
    create_consent_path,
    create_study_path,
    new_session_id,
    now_iso,
    save_consent,
    save_study,
)


STUDY_MODEL_REGISTRY = {
    "gamma": {
        "stochastic": "movingout_gamma_stochastic",
    },
    "comedi": {
        "stochastic": "movingout_comedi_stochastic",
    },
    "prism": {
        "stochastic": "movingout_prism_stochastic",
    },
}

STUDY_SEQUENCE: Tuple[Tuple[str, str, str], ...] = (
    ("gamma", "stochastic", "EfficientRoutes"),
    ("gamma", "stochastic", "SequentialRotations"),
    ("comedi", "stochastic", "EfficientRoutes"),
    ("comedi", "stochastic", "SequentialRotations"),
    ("prism", "stochastic", "EfficientRoutes"),
    ("prism", "stochastic", "SequentialRotations"),
)

TUTORIAL_JOINT_STEP_LIMIT = 3000
FORMAL_JOINT_STEP_LIMIT = 500
TUTORIAL_SIDEBAR_WIDTH = 320
TUTORIAL_SKIP_JOYSTICK_BUTTONS = {2, 3}
CONSENT_SCROLL_DEADZONE = 0.25
CONSENT_STICK_SCROLL_PER_FRAME = 18.0


@dataclass(frozen=True)
class ConsentParagraph:
    text: str
    kind: str


@dataclass(frozen=True)
class ConsentLine:
    text: str
    font: pygame.font.Font
    color: Tuple[int, int, int]
    x: int
    y: int
    centered: bool = False


class StudyCollectionApp(CollectionApp):
    """Run one consented tutorial plus six formal study episodes."""

    def __init__(
        self,
        output_dir: str = "saved_datasets",
        ai_seed: Optional[int] = None,
        ai_timeout: float = 2.0,
        input_mode: str = "joystick",
        human_agent: str = "blue",
        questions: Optional[Sequence[QuestionnaireItem]] = None,
        consent_form_path: Optional[str] = None,
    ) -> None:
        if human_agent not in {"blue", "red"}:
            raise ValueError("human_agent must be 'blue' or 'red'.")

        super().__init__(
            output_dir=output_dir,
            ai_seed=ai_seed,
            ai_timeout=ai_timeout,
            input_mode=input_mode,
            questions=questions,
        )
        self.study_human_agent = human_agent
        repository_root = Path(__file__).resolve().parents[2]
        self.consent_form_path = Path(
            consent_form_path
            or repository_root / "#6932 IRB-SBS Consent Form.docx"
        ).resolve()

        # The formal CONFIG page only retains participant identity fields.
        self.start_button.rect = pygame.Rect(330, 285, 240, 58)

        self.consent_viewport = pygame.Rect(45, 75, 800, 760)
        self.consent_scroll = 0.0
        self.consent_virtual_height = 0
        self.consent_lines: List[ConsentLine] = []
        self.consent_stick_y = 0.0
        self.consent_dragging = False
        self.consent_drag_offset = 0
        self.consent_agree_virtual_y = 0
        self.consent_document_sha256 = ""
        self.agree_button = Button(
            pygame.Rect(335, 0, 230, 56),
            "Agree",
            self.heading_font,
            base_color=(35, 135, 65),
            hover_color=(55, 165, 85),
        )

        self.consent_id: Optional[str] = None
        self.consent_path: Optional[str] = None
        self.study_path: Optional[str] = None
        self.study_record: Optional[StudyRecord] = None
        self.study_ai_seed: Optional[int] = None
        self.current_episode_index = -1
        self.current_episode: Optional[StudyEpisode] = None
        self.current_questionnaire: Optional[StudyQuestionnaire] = None
        self.current_questionnaire_number = 0

        self.tutorial_stage = 0
        self.tutorial_start_position: Optional[Tuple[float, float]] = None
        self.tutorial_next_control_at = 0.0
        self.tutorial_joint_steps = 0
        self.tutorial_started_at: Optional[str] = None
        self.tutorial_started_monotonic: Optional[float] = None

    @property
    def study_ai_agent(self) -> str:
        return "red" if self.study_human_agent == "blue" else "blue"

    def _config_errors(self) -> Dict[str, str]:
        errors: Dict[str, str] = {}
        if not self.computing_id_input.text.strip():
            errors["computing_id"] = "Computing ID is required."
        if not validate_email(self.email_input.text):
            errors["email"] = "Enter a valid email address."
        return errors

    def _handle_config_events(
        self, events: Sequence[pygame.event.Event]
    ) -> None:
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                if self.computing_id_input.active:
                    self.computing_id_input.active = False
                    self.email_input.active = True
                else:
                    self.email_input.active = False
                    self.computing_id_input.active = True
                continue
            self.computing_id_input.handle_event(event)
            self.email_input.handle_event(event)
            if self.start_button.is_clicked(event):
                errors = self._config_errors()
                if errors:
                    self.error_message = next(iter(errors.values()))
                elif (
                    self.input_mode == "joystick"
                    and pygame.joystick.get_count() < 1
                ):
                    self.error_message = (
                        "No joystick detected. Connect one and try again."
                    )
                else:
                    self.error_message = ""
                    self._enter_consent()
                return

    def _draw_config(self) -> None:
        self.screen.fill((255, 255, 255))
        title = self.title_font.render("Data Collection", True, (20, 20, 25))
        self.screen.blit(
            title, title.get_rect(center=(WINDOW_WIDTH // 2, 48))
        )
        errors = self._config_errors()

        self._draw_label("Computing ID", 112)
        self.computing_id_input.draw(
            self.screen,
            invalid=bool(
                self.computing_id_input.text
                and errors.get("computing_id")
            ),
        )
        self._draw_label("Email", 188)
        self.email_input.draw(
            self.screen,
            invalid=bool(
                self.email_input.text and errors.get("email")
            ),
        )
        self.start_button.draw(self.screen, enabled=not bool(errors))
        if self.error_message:
            error = self.error_font.render(
                self.error_message, True, (220, 65, 65)
            )
            self.screen.blit(
                error,
                error.get_rect(center=(WINDOW_WIDTH // 2, 430)),
            )

    def _enter_consent(self) -> None:
        try:
            paragraphs = self._read_consent_document()
            self._layout_consent_document(paragraphs)
            if self.input_mode == "joystick":
                self.controller = JoystickController()
            else:
                self.controller = KeyboardController()
        except Exception as exc:
            self.state = "CONFIG"
            self.error_message = f"Could not load consent form: {exc}"
            return

        self._set_main_display("Moving Out Informed Consent")
        self.consent_scroll = 0.0
        self.consent_stick_y = 0.0
        self.consent_dragging = False
        self._position_consent_widgets()
        self.state = "CONSENT"

    def _read_consent_document(self) -> List[ConsentParagraph]:
        if not self.consent_form_path.is_file():
            raise FileNotFoundError(
                f"Consent document not found: {self.consent_form_path}"
            )
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError(
                "python-docx is required; install requirements.txt."
            ) from exc

        raw_bytes = self.consent_form_path.read_bytes()
        self.consent_document_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        document = Document(str(self.consent_form_path))
        result: List[ConsentParagraph] = []
        for index, paragraph in enumerate(document.paragraphs):
            text = paragraph.text.strip()
            if text.startswith("Print Name:") or text.startswith("Signature:"):
                continue
            if not text:
                result.append(ConsentParagraph("", "blank"))
                continue

            paragraph_properties = paragraph._p.pPr
            is_list = bool(
                paragraph_properties is not None
                and paragraph_properties.numPr is not None
            )
            style_name = (
                paragraph.style.name.lower()
                if paragraph.style is not None
                else ""
            )
            if index == 0:
                kind = "title"
            elif is_list:
                kind = "bullet"
            elif style_name.startswith("heading") or text == "Agreement:":
                kind = "heading"
            else:
                kind = "body"
            result.append(ConsentParagraph(text, kind))
        if not any(paragraph.text for paragraph in result):
            raise ValueError("Consent document contains no readable text.")
        return result

    def _layout_consent_document(
        self, paragraphs: Sequence[ConsentParagraph]
    ) -> None:
        self.consent_lines = []
        content_width = self.consent_viewport.width - 90
        y = 25
        for paragraph in paragraphs:
            if paragraph.kind == "blank":
                y += 8
                continue
            if paragraph.kind == "title":
                font = self.heading_font
                color = (20, 20, 25)
                centered = True
                prefix = ""
                after = 22
            elif paragraph.kind == "heading":
                font = self.body_font
                color = (20, 20, 25)
                centered = False
                prefix = ""
                after = 12
            else:
                font = self.small_font
                color = (35, 35, 40)
                centered = False
                prefix = "- " if paragraph.kind == "bullet" else ""
                after = 13

            lines = self._wrap_text(
                prefix + paragraph.text,
                font,
                content_width,
            )
            for line in lines:
                self.consent_lines.append(
                    ConsentLine(
                        line,
                        font,
                        color,
                        40,
                        y,
                        centered=centered,
                    )
                )
                y += font.get_linesize() + 3
            y += after

        self.consent_agree_virtual_y = y + 30
        self.consent_virtual_height = y + 120
        self._position_consent_widgets()

    @property
    def _consent_max_scroll(self) -> float:
        return float(
            max(
                0,
                self.consent_virtual_height
                - self.consent_viewport.height,
            )
        )

    def _clamp_consent_scroll(self) -> None:
        self.consent_scroll = max(
            0.0,
            min(self._consent_max_scroll, self.consent_scroll),
        )

    def _position_consent_widgets(self) -> None:
        agree_y = int(
            self.consent_viewport.y
            + self.consent_agree_virtual_y
            - self.consent_scroll
        )
        self.agree_button.rect = pygame.Rect(
            self.consent_viewport.centerx - 115,
            agree_y,
            230,
            56,
        )

    def _consent_scrollbar_geometry(
        self,
    ) -> Tuple[pygame.Rect, pygame.Rect]:
        track = pygame.Rect(
            self.consent_viewport.right + 8,
            self.consent_viewport.y,
            12,
            self.consent_viewport.height,
        )
        if self.consent_virtual_height <= 0:
            return track, track.copy()
        ratio = min(
            1.0,
            self.consent_viewport.height
            / float(self.consent_virtual_height),
        )
        thumb_height = max(48, int(track.height * ratio))
        travel = track.height - thumb_height
        position_ratio = (
            self.consent_scroll / self._consent_max_scroll
            if self._consent_max_scroll > 0
            else 0.0
        )
        thumb = pygame.Rect(
            track.x,
            track.y + int(travel * position_ratio),
            track.width,
            thumb_height,
        )
        return track, thumb

    def _set_consent_scroll_from_thumb(self, thumb_top: int) -> None:
        track, thumb = self._consent_scrollbar_geometry()
        travel = track.height - thumb.height
        if travel <= 0:
            self.consent_scroll = 0.0
            return
        clamped_top = max(track.top, min(track.bottom - thumb.height, thumb_top))
        ratio = (clamped_top - track.top) / float(travel)
        self.consent_scroll = ratio * self._consent_max_scroll
        self._clamp_consent_scroll()
        self._position_consent_widgets()

    def _consent_can_agree(self) -> bool:
        return (
            self._consent_max_scroll - self.consent_scroll <= 3.0
        )

    def _handle_consent_events(
        self, events: Sequence[pygame.event.Event]
    ) -> None:
        self._position_consent_widgets()
        track, thumb = self._consent_scrollbar_geometry()
        for event in events:
            if (
                event.type == pygame.JOYAXISMOTION
                and getattr(event, "joy", None) == 0
                and event.axis == 1
            ):
                self.consent_stick_y = float(event.value)
                continue
            if event.type == pygame.MOUSEWHEEL:
                self.consent_scroll -= float(event.y) * 55.0
                self._clamp_consent_scroll()
                self._position_consent_widgets()
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.consent_scroll -= 35
                elif event.key == pygame.K_DOWN:
                    self.consent_scroll += 35
                elif event.key == pygame.K_PAGEUP:
                    self.consent_scroll -= self.consent_viewport.height * 0.8
                elif event.key == pygame.K_PAGEDOWN:
                    self.consent_scroll += self.consent_viewport.height * 0.8
                elif event.key == pygame.K_HOME:
                    self.consent_scroll = 0
                elif event.key == pygame.K_END:
                    self.consent_scroll = self._consent_max_scroll
                self._clamp_consent_scroll()
                self._position_consent_widgets()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if thumb.collidepoint(event.pos):
                    self.consent_dragging = True
                    self.consent_drag_offset = event.pos[1] - thumb.top
                    continue
                if track.collidepoint(event.pos):
                    self._set_consent_scroll_from_thumb(
                        event.pos[1] - thumb.height // 2
                    )
                    continue
            elif event.type == pygame.MOUSEMOTION and self.consent_dragging:
                self._set_consent_scroll_from_thumb(
                    event.pos[1] - self.consent_drag_offset
                )
                continue
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.consent_dragging = False

            if self.agree_button.is_clicked(
                event, enabled=self._consent_can_agree()
            ):
                self._agree_to_consent()
                return

    def _update_consent_scroll(self) -> None:
        if abs(self.consent_stick_y) <= CONSENT_SCROLL_DEADZONE:
            return
        normalized = (
            abs(self.consent_stick_y) - CONSENT_SCROLL_DEADZONE
        ) / (1.0 - CONSENT_SCROLL_DEADZONE)
        self.consent_scroll += (
            math.copysign(normalized, self.consent_stick_y)
            * CONSENT_STICK_SCROLL_PER_FRAME
        )
        self._clamp_consent_scroll()
        self._position_consent_widgets()

    def _draw_consent(self) -> None:
        self.screen.fill((255, 255, 255))
        heading = self.title_font.render(
            "Informed Consent", True, (20, 20, 25)
        )
        self.screen.blit(
            heading,
            heading.get_rect(center=(WINDOW_WIDTH // 2, 38)),
        )
        pygame.draw.rect(
            self.screen,
            (250, 250, 252),
            self.consent_viewport,
            border_radius=6,
        )
        pygame.draw.rect(
            self.screen,
            (160, 160, 168),
            self.consent_viewport,
            1,
            border_radius=6,
        )

        previous_clip = self.screen.get_clip()
        self.screen.set_clip(self.consent_viewport)
        for line in self.consent_lines:
            rendered = line.font.render(line.text, True, line.color)
            y = int(
                self.consent_viewport.y + line.y - self.consent_scroll
            )
            if line.centered:
                rect = rendered.get_rect(
                    centerx=self.consent_viewport.centerx,
                    top=y,
                )
            else:
                rect = rendered.get_rect(
                    left=self.consent_viewport.x + line.x,
                    top=y,
                )
            self.screen.blit(rendered, rect)

        self.agree_button.draw(
            self.screen,
            enabled=self._consent_can_agree(),
        )
        self.screen.set_clip(previous_clip)

        track, thumb = self._consent_scrollbar_geometry()
        pygame.draw.rect(self.screen, (225, 225, 230), track, border_radius=6)
        pygame.draw.rect(self.screen, (105, 110, 125), thumb, border_radius=6)
        hint_text = (
            "Scroll: mouse wheel, arrow/Page keys, or joystick left stick"
            if self.input_mode == "joystick"
            else "Scroll: mouse wheel or arrow/Page keys"
        )
        hint = self.small_font.render(hint_text, True, (85, 85, 92))
        self.screen.blit(
            hint,
            hint.get_rect(center=(WINDOW_WIDTH // 2, 865)),
        )

    def _agree_to_consent(self) -> None:
        if not self._consent_can_agree():
            return
        agreed_at = now_iso()
        consent_id = new_session_id()
        session_id = new_session_id()
        consent_record = ConsentRecord(
            consent_id=consent_id,
            computing_id=self.computing_id_input.text.strip(),
            email=self.email_input.text.strip(),
            agreed_at=agreed_at,
            document_name=self.consent_form_path.name,
            document_sha256=self.consent_document_sha256,
        )
        consent_path = create_consent_path(self.output_dir, consent_id)
        study_path = create_study_path(self.output_dir, session_id)
        ai_seed = (
            self.requested_ai_seed
            if self.requested_ai_seed is not None
            else secrets.randbelow(2**31)
        )
        study_record = StudyRecord(
            metadata={
                "session_id": session_id,
                "computing_id": self.computing_id_input.text.strip(),
                "email": self.email_input.text.strip(),
                "human_agent": self.study_human_agent,
                "human_agent_index": (
                    0 if self.study_human_agent == "blue" else 1
                ),
                "ai_agent": self.study_ai_agent,
                "ai_agent_index": (
                    0 if self.study_ai_agent == "blue" else 1
                ),
                "model_schedule": [
                    "gamma_stochastic",
                    "comedi_stochastic",
                    "prism_stochastic",
                ],
                "input_mode": self.input_mode,
                "base_ai_seed": ai_seed,
                "consent_id": consent_id,
                "consent_agreed_at": agreed_at,
                "consent_document_name": self.consent_form_path.name,
                "consent_document_sha256": (
                    self.consent_document_sha256
                ),
                "created_at": agreed_at,
                "started_at": None,
                "ended_at": None,
                "end_reason": None,
                "session_status": "tutorial_pending",
                "current_stage": "tutorial",
                "formal_episode_count": len(STUDY_SEQUENCE),
                "questionnaire_count": len(STUDY_MODEL_REGISTRY),
            },
            tutorial={
                "status": "pending",
                "started_at": None,
                "ended_at": None,
                "end_reason": None,
                "joint_steps": 0,
                "joint_step_limit": TUTORIAL_JOINT_STEP_LIMIT,
            },
        )
        try:
            save_consent(consent_path, consent_record)
            save_study(study_path, study_record)
        except Exception as exc:
            self.state = "ERROR"
            self.error_message = f"Could not save consent: {exc}"
            return

        self.consent_id = consent_id
        self.consent_path = consent_path
        self.study_path = study_path
        self.session_path = study_path
        self.study_ai_seed = ai_seed
        self.study_record = study_record
        self._prepare_tutorial()

    def _reset_controller_for_stage(self) -> None:
        if self.controller is None:
            if self.input_mode == "joystick":
                self.controller = JoystickController()
            else:
                self.controller = KeyboardController()
        reset = getattr(self.controller, "reset", None)
        if callable(reset):
            reset()
        self.input_started = False

    def _prepare_tutorial(self) -> None:
        assert self.study_record is not None
        try:
            self._close_ai()
            self._close_environment()
            self._reset_controller_for_stage()
            self.env = MovingOutEnv(
                use_state=False,
                map_name="Simple",
                max_episode_steps=None,
            )
            self.config = CollectionConfig(
                computing_id=self.computing_id_input.text.strip(),
                email=self.email_input.text.strip(),
                human_agent=self.study_human_agent,
                ai_model="tutorial_idle",
                map_id="Simple",
                input_mode=self.input_mode,
            )
            self.frame = self.env.render("rgb_array")
            self._set_tutorial_display()
            robot_key = (
                "robot_1"
                if self.study_human_agent == "blue"
                else "robot_2"
            )
            position = self.env.get_all_states()[robot_key]["pos"]
            self.tutorial_start_position = (
                float(position[0]),
                float(position[1]),
            )
            self.tutorial_stage = 0
            self.tutorial_joint_steps = 0
            self.tutorial_started_at = None
            self.tutorial_started_monotonic = None
            self.score = 0.0
            self.study_record.tutorial.update(
                {
                    "status": "waiting_for_input",
                    "started_at": None,
                    "ended_at": None,
                    "end_reason": None,
                    "joint_steps": 0,
                }
            )
            self.study_record.metadata.update(
                {
                    "session_status": "tutorial_pending",
                    "current_stage": "tutorial",
                }
            )
            self._save_study()
            self.state = "TUTORIAL_WAITING"
        except Exception as exc:
            self._abort_study("tutorial_startup_error", str(exc))

    def _set_tutorial_display(self) -> None:
        if self.frame is None:
            raise RuntimeError("Tutorial frame is not initialized.")
        combined = self.overlay_renderer.display_image_with_overlay(
            self.frame, 0.0, 0.0
        )
        size = (
            TUTORIAL_SIDEBAR_WIDTH + combined.shape[1],
            combined.shape[0],
        )
        if self.screen.get_size() != size:
            self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("MovingOut Tutorial")

    def _tutorial_skip_requested(
        self, events: Sequence[pygame.event.Event]
    ) -> bool:
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                return True
            if (
                event.type == pygame.JOYBUTTONDOWN
                and getattr(event, "joy", None) == 0
                and event.button in TUTORIAL_SKIP_JOYSTICK_BUTTONS
            ):
                return True
        return False

    def _handle_tutorial_events(
        self, events: Sequence[pygame.event.Event]
    ) -> None:
        if self._tutorial_skip_requested(events):
            self._finish_tutorial("user_skipped")
            return
        self._handle_controller_events(events)
        if (
            self.state == "TUTORIAL_WAITING"
            and self._input_has_started()
        ):
            self._start_tutorial_after_input()

    def _start_tutorial_after_input(self) -> None:
        assert self.study_record is not None
        started_at = now_iso()
        started_monotonic = time.monotonic()
        self.tutorial_started_at = started_at
        self.tutorial_started_monotonic = started_monotonic
        self.tutorial_next_control_at = started_monotonic
        self.study_record.tutorial.update(
            {
                "status": "running",
                "started_at": started_at,
            }
        )
        self.study_record.metadata["session_status"] = "tutorial_running"
        self._save_study()
        self.state = "TUTORIAL"

    @staticmethod
    def _distance(
        first: Sequence[float], second: Sequence[float]
    ) -> float:
        return math.hypot(
            float(first[0]) - float(second[0]),
            float(first[1]) - float(second[1]),
        )

    def _update_tutorial_stage(
        self,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        human_action: Sequence[Any],
    ) -> None:
        robot_key = (
            "robot_1" if self.study_human_agent == "blue" else "robot_2"
        )
        before_robot = state_before[robot_key]
        after_robot = state_after[robot_key]
        if (
            self.tutorial_stage == 0
            and self.tutorial_start_position is not None
            and self._distance(
                after_robot["pos"],
                self.tutorial_start_position,
            )
            >= 0.05
        ):
            self.tutorial_stage = 1
        if (
            self.tutorial_stage == 1
            and after_robot.get("holded_item_id") is not None
        ):
            self.tutorial_stage = 2
        if (
            self.tutorial_stage == 2
            and float(human_action[0]) < -0.2
            and self._distance(
                before_robot["pos"],
                after_robot["pos"],
            )
            >= 0.005
        ):
            self.tutorial_stage = 3

    def _update_tutorial(self) -> None:
        assert self.env is not None
        assert self.controller is not None
        assert self.config is not None
        now = time.monotonic()
        if now < self.tutorial_next_control_at:
            return
        try:
            state_before = self.env.get_all_states()
            human_action = self._get_human_action()
            joint_action = compose_joint_action(
                human_action,
                IDLE_AI_ACTION,
                self.study_human_agent,
            )
            _, _, terminated, truncated, info = self.env.step(joint_action)
            state_after = self.env.get_all_states()
            self.frame = self.env.render("rgb_array")
            self.tutorial_joint_steps += 1
            self.score = max(
                0.0,
                min(1.0, float(info.get("eval_score", 0.0))),
            )
            self._update_tutorial_stage(
                state_before,
                state_after,
                human_action,
            )
            self.tutorial_next_control_at = max(
                self.tutorial_next_control_at + CONTROL_INTERVAL,
                time.monotonic(),
            )

            if terminated or self.score >= 1.0:
                self.tutorial_stage = len(self._tutorial_instructions())
                self._finish_tutorial("completed")
            elif info.get("physics_nan"):
                self._abort_study(
                    "environment_error",
                    "Tutorial environment physics failed.",
                )
            elif (
                truncated
                or self.tutorial_joint_steps
                >= TUTORIAL_JOINT_STEP_LIMIT
            ):
                self._finish_tutorial("step_limit")
        except Exception as exc:
            self._abort_study("tutorial_runtime_error", str(exc))

    def _tutorial_instructions(self) -> List[str]:
        if self.input_mode == "joystick":
            return [
                "Use the left stick to move up, down, left, and right.",
                "Press the R button to grab or release an object.",
                "Hold ZL and move the left stick to move backward.",
                "Move all objects into the target area to finish.",
            ]
        return [
            "Use W, A, S, and D to move.",
            "Press Space to grab or release an object.",
            "Hold H and press W, A, S, or D to move backward.",
            "Move all objects into the target area to finish.",
        ]

    def _draw_tutorial(self) -> None:
        if self.frame is None:
            self.screen.fill((0, 0, 0))
            return
        elapsed = 0.0
        if self.tutorial_started_monotonic is not None:
            elapsed = max(
                0.0,
                time.monotonic() - self.tutorial_started_monotonic,
            )
        combined = self.overlay_renderer.display_image_with_overlay(
            self.frame,
            self.score,
            elapsed,
        )
        game_surface = pygame.surfarray.make_surface(
            combined.swapaxes(0, 1)
        )
        self.screen.fill((245, 245, 248))
        self.screen.blit(game_surface, (TUTORIAL_SIDEBAR_WIDTH, 0))
        pygame.draw.line(
            self.screen,
            (155, 155, 165),
            (TUTORIAL_SIDEBAR_WIDTH - 1, 0),
            (TUTORIAL_SIDEBAR_WIDTH - 1, self.screen.get_height()),
            2,
        )
        title = self.title_font.render("Tutorial", True, (20, 20, 25))
        self.screen.blit(title, (28, 28))

        y = 95
        for index, instruction in enumerate(self._tutorial_instructions()):
            if index < self.tutorial_stage:
                prefix = "[Done]"
                color = (35, 130, 65)
            elif index == self.tutorial_stage:
                prefix = "[Now]"
                color = (35, 85, 175)
            else:
                prefix = "[Next]"
                color = (120, 120, 128)
            prefix_surface = self.small_font.render(prefix, True, color)
            self.screen.blit(prefix_surface, (28, y))
            y += prefix_surface.get_height() + 4
            for line in self._wrap_text(
                instruction,
                self.small_font,
                TUTORIAL_SIDEBAR_WIDTH - 56,
            ):
                rendered = self.small_font.render(line, True, color)
                self.screen.blit(rendered, (28, y))
                y += rendered.get_height() + 2
            y += 18

        skip_text = (
            "Press controller X to skip"
            if self.input_mode == "joystick"
            else "Press keyboard X to skip"
        )
        skip = self.small_font.render(skip_text, True, (115, 65, 65))
        self.screen.blit(
            skip,
            skip.get_rect(
                center=(
                    TUTORIAL_SIDEBAR_WIDTH // 2,
                    self.screen.get_height() - 35,
                )
            ),
        )

    def _finish_tutorial(self, reason: str) -> None:
        assert self.study_record is not None
        status = {
            "completed": "completed",
            "user_skipped": "skipped",
            "step_limit": "step_limit",
        }[reason]
        self.study_record.tutorial.update(
            {
                "status": status,
                "ended_at": now_iso(),
                "end_reason": reason,
                "joint_steps": self.tutorial_joint_steps,
                "completed_instruction_count": self.tutorial_stage,
            }
        )
        self.study_record.metadata.update(
            {
                "session_status": "formal_pending",
                "current_stage": "episode_1",
            }
        )
        self._close_environment()
        self._save_study()
        self._prepare_formal_episode(0)

    @staticmethod
    def _formal_model_name(model_family: str, condition: str) -> str:
        return STUDY_MODEL_REGISTRY[model_family][condition]

    def _prepare_formal_episode(self, episode_index: int) -> None:
        assert self.study_record is not None
        assert self.study_ai_seed is not None
        model_family, condition, map_id = STUDY_SEQUENCE[episode_index]
        model_name = self._formal_model_name(model_family, condition)
        ai_seed = self.study_ai_seed + episode_index
        config = CollectionConfig(
            computing_id=self.computing_id_input.text.strip(),
            email=self.email_input.text.strip(),
            human_agent=self.study_human_agent,
            ai_model=model_name,
            map_id=map_id,
            input_mode=self.input_mode,
        )
        model_class = MODEL_REGISTRY[model_name]
        episode = StudyEpisode(
            metadata={
                "episode_index": episode_index + 1,
                "model_family": model_family,
                "condition": condition,
                "map_id": map_id,
                "human_agent": config.human_agent,
                "human_agent_index": config.human_agent_index,
                "ai_agent": config.ai_agent,
                "ai_agent_index": config.ai_agent_index,
                "ai_model": model_name,
                "ai_seed": ai_seed,
                "input_mode": self.input_mode,
                "created_at": now_iso(),
                "started_at": None,
                "ended_at": None,
                "end_reason": None,
                "status": "starting",
                "joint_step_limit": FORMAL_JOINT_STEP_LIMIT,
                "control_hz": 1.0 / CONTROL_INTERVAL,
                "async_ai": True,
                "ai_plan_horizon": model_class.plan_horizon,
                "ai_preferred_action_count": (
                    model_class.preferred_action_count
                ),
                "ai_fallback_policy": "queued_plan_then_idle",
                "ai_plan_alignment_policy": "rebase_from_first_action",
            }
        )
        self.study_record.episodes.append(episode)
        self.current_episode_index = episode_index
        self.current_episode = episode
        self.record = episode
        self.config = config
        self.study_record.metadata.update(
            {
                "current_stage": f"episode_{episode_index + 1}",
                "session_status": "episode_starting",
            }
        )
        self._set_main_display("Moving Out Data Collection")
        self.state = "AI_STARTING"
        self._draw_starting()
        pygame.display.flip()
        self._save_study()

        try:
            self._close_ai()
            self._close_environment()
            self._reset_controller_for_stage()
            self.env = MovingOutEnv(
                use_state=False,
                map_name=map_id,
                max_episode_steps=None,
            )
            model_metadata = model_class.session_metadata(
                map_id,
                config.ai_agent_index,
            )
            response_timeout = max(
                self.ai_timeout,
                model_class.minimum_response_timeout,
            )
            self.ai_client = AIProcessClient(
                response_timeout=response_timeout
            )
            self.ai_client.start(
                model_name=model_name,
                agent_index=config.ai_agent_index,
                seed=ai_seed,
                startup_timeout=model_class.startup_timeout,
                map_id=map_id,
            )
            self.frame = self.env.render("rgb_array")
            self._set_game_display()
            episode.metadata.update(
                {
                    "status": "waiting_for_input",
                    **model_metadata,
                }
            )
            self.active_action_buffer = []
            self.next_ai_request_id = 1
            self.last_requested_source_step = None
            self.joint_steps = 0
            self.score = 0.0
            self.input_started = False
            self.study_record.metadata["session_status"] = (
                "waiting_for_input"
            )
            self._save_study()
            self.state = "WAITING_FOR_INPUT"
        except Exception as exc:
            episode.metadata.update(
                {
                    "status": "aborted",
                    "ended_at": now_iso(),
                    "end_reason": "startup_error",
                    "error": str(exc),
                }
            )
            self._abort_study("ai_startup_error", str(exc))

    def _start_game_after_input(self) -> None:
        super()._start_game_after_input()
        assert self.current_episode is not None
        assert self.study_record is not None
        self.current_episode.metadata["status"] = "running"
        self.study_record.metadata.update(
            {
                "started_at": (
                    self.study_record.metadata["started_at"]
                    or self.current_episode.metadata["started_at"]
                ),
                "session_status": "episode_running",
            }
        )
        self._save_study()

    def _update_game(self) -> None:
        """Run one asynchronous control update and enforce joint-step limits."""

        episode = self.current_episode
        super()._update_game()
        if (
            self.state == "GAME"
            and self.current_episode is episode
            and self.joint_steps >= FORMAL_JOINT_STEP_LIMIT
        ):
            self._finish_episode("step_limit")

    def _finish_episode(self, reason: str) -> None:
        assert self.current_episode is not None
        assert self.study_record is not None
        if reason == "time_limit":
            reason = "step_limit"
        self.current_episode.metadata.update(
            {
                "ended_at": now_iso(),
                "end_reason": reason,
                "status": "complete",
                "joint_steps": self.joint_steps,
                "environment_steps": int(
                    getattr(self.env, "_episode_steps", 0)
                ),
                "final_score": self.score,
            }
        )
        self._close_ai()
        self._close_environment()
        self.study_record.metadata["session_status"] = "between_stages"
        self._save_study()

        if self.current_episode_index in {1, 3, 5}:
            model_family = self.current_episode.metadata["model_family"]
            questionnaire_number = self.current_episode_index // 2 + 1
            self._begin_questionnaire(
                model_family,
                questionnaire_number,
            )
        else:
            self._prepare_formal_episode(self.current_episode_index + 1)

    def _begin_questionnaire(
        self, model_family: str, questionnaire_number: int
    ) -> None:
        assert self.study_record is not None
        self.current_episode = None
        questionnaire = StudyQuestionnaire(
            metadata={
                "questionnaire_index": questionnaire_number,
                "model_family": model_family,
                "condition": "stochastic",
                "status": "in_progress",
                "started_at": now_iso(),
                "completed_at": None,
            },
            items=[
                StudyQuestionnaireItem(
                    question=item.question,
                    options=list(item.options),
                )
                for item in self.question_templates
            ],
        )
        self.study_record.questionnaires.append(questionnaire)
        self.current_questionnaire = questionnaire
        self.current_questionnaire_number = questionnaire_number
        self.record = SessionRecord(
            metadata={},
            questionnaire=[
                QuestionnaireItem(
                    item.question,
                    list(item.options),
                )
                for item in self.question_templates
            ],
        )
        self.survey_index = 0
        self.study_record.metadata.update(
            {
                "current_stage": (
                    f"questionnaire_{questionnaire_number}"
                ),
                "session_status": "awaiting_questionnaire",
            }
        )
        self._set_main_display("Moving Out Questionnaire")
        self._save_study()
        self.state = "SURVEY"

    def _handle_survey_events(
        self, events: Sequence[pygame.event.Event]
    ) -> None:
        assert isinstance(self.record, SessionRecord)
        assert self.current_questionnaire is not None
        question = self.record.questionnaire[self.survey_index]
        response = self.current_questionnaire.items[self.survey_index]
        next_rect = pygame.Rect(350, 690, 200, 56)
        skip_rect = pygame.Rect(585, 690, 150, 56)
        for event in events:
            if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
                continue
            for option, rect in self._survey_option_rects():
                if rect.collidepoint(event.pos):
                    question.answer = option
                    response.answer = option
                    response.skipped = False
                    break
            if next_rect.collidepoint(event.pos) and question.answer is not None:
                self._advance_questionnaire()
                return
            if skip_rect.collidepoint(event.pos):
                question.answer = None
                response.answer = None
                response.skipped = True
                self._advance_questionnaire()
                return

    def _advance_questionnaire(self) -> None:
        assert isinstance(self.record, SessionRecord)
        assert self.current_questionnaire is not None
        self._save_study()
        if self.survey_index + 1 < len(self.record.questionnaire):
            self.survey_index += 1
            return
        self.current_questionnaire.metadata.update(
            {
                "status": "complete",
                "completed_at": now_iso(),
            }
        )
        self._save_study()
        if self.current_questionnaire_number < len(STUDY_MODEL_REGISTRY):
            self.current_questionnaire = None
            self._prepare_formal_episode(
                self.current_questionnaire_number * 2
            )
            return
        self._complete_study()

    def _draw_survey(self) -> None:
        super()._draw_survey()
        skip_button = Button(
            pygame.Rect(585, 690, 150, 56),
            "Skip",
            self.heading_font,
            base_color=(105, 105, 115),
            hover_color=(130, 130, 142),
        )
        skip_button.draw(self.screen)

    def _complete_study(self) -> None:
        assert self.study_record is not None
        self.study_record.metadata.update(
            {
                "ended_at": now_iso(),
                "end_reason": "completed",
                "session_status": "complete",
                "current_stage": "complete",
            }
        )
        self._save_study()
        self.state = "COMPLETE"
        self.running = False

    def _save_study(self) -> None:
        if self.study_record is None or self.study_path is None:
            raise RuntimeError("No active study record to save.")
        save_study(self.study_path, self.study_record)

    def _save_record(self) -> None:
        self._save_study()

    def _abort_session(self, reason: str, message: str) -> None:
        self._abort_study(reason, message)

    def _abort_study(self, reason: str, message: str) -> None:
        environment_steps = int(getattr(self.env, "_episode_steps", 0))
        self._close_ai()
        self._close_environment()
        self._set_main_display("Moving Out Data Collection")
        save_error = ""
        if (
            self.current_episode is not None
            and self.current_episode.metadata.get("status") != "complete"
        ):
            self.current_episode.metadata.update(
                {
                    "ended_at": (
                        self.current_episode.metadata.get("ended_at")
                        or now_iso()
                    ),
                    "end_reason": reason,
                    "status": "aborted",
                    "error": message,
                    "joint_steps": self.joint_steps,
                    "environment_steps": environment_steps,
                }
            )
        if self.study_record is not None:
            self.study_record.metadata.update(
                {
                    "ended_at": now_iso(),
                    "end_reason": reason,
                    "session_status": "aborted",
                    "error": message,
                }
            )
            try:
                self._save_study()
            except Exception as exc:
                save_error = f" Saving also failed: {exc}"
        self.error_message = message + save_error
        self.state = "ERROR"

    def _handle_close(self) -> None:
        self._close_ai()
        self._close_environment()
        if self.study_record is not None:
            self.study_record.metadata.update(
                {
                    "ended_at": now_iso(),
                    "end_reason": "user_closed",
                    "session_status": "aborted",
                }
            )
            if self.current_questionnaire is not None:
                self.current_questionnaire.metadata["status"] = "incomplete"
            try:
                self._save_study()
            except Exception:
                pass
        self.running = False

    def run(self) -> None:
        try:
            while self.running:
                events = pygame.event.get()
                if any(event.type == pygame.QUIT for event in events):
                    self._handle_close()
                    continue

                if self.state == "CONFIG":
                    self._handle_config_events(events)
                elif self.state == "CONSENT":
                    self._handle_consent_events(events)
                    self._update_consent_scroll()
                elif self.state in {"TUTORIAL_WAITING", "TUTORIAL"}:
                    self._handle_tutorial_events(events)
                    if self.state == "TUTORIAL":
                        self._update_tutorial()
                elif self.state == "WAITING_FOR_INPUT":
                    self._handle_controller_events(events)
                    if self._input_has_started():
                        self._start_game_after_input()
                elif self.state == "GAME":
                    self._handle_controller_events(events)
                    self._update_game()
                elif self.state == "SURVEY":
                    self._handle_survey_events(events)
                elif self.state == "ERROR":
                    for event in events:
                        if self.exit_button.is_clicked(event):
                            self.running = False

                if self.state == "CONFIG":
                    self._draw_config()
                elif self.state == "CONSENT":
                    self._draw_consent()
                elif self.state == "AI_STARTING":
                    self._draw_starting()
                elif self.state in {"TUTORIAL_WAITING", "TUTORIAL"}:
                    self._draw_tutorial()
                elif self.state == "WAITING_FOR_INPUT":
                    self._draw_game(waiting=True)
                elif self.state == "GAME":
                    self._draw_game()
                elif self.state == "SURVEY":
                    self._draw_survey()
                elif self.state == "ERROR":
                    self._draw_error()

                pygame.display.flip()
                self.clock.tick(UI_FPS)
        finally:
            self._close_ai()
            self._close_environment()
            pygame.key.stop_text_input()
            pygame.quit()


def run_study_collection_app(
    output_dir: str = "saved_datasets",
    ai_seed: Optional[int] = None,
    ai_timeout: float = 2.0,
    input_mode: str = "joystick",
    human_agent: str = "blue",
    questions: Optional[Sequence[QuestionnaireItem]] = None,
    consent_form_path: Optional[str] = None,
) -> None:
    app = StudyCollectionApp(
        output_dir=output_dir,
        ai_seed=ai_seed,
        ai_timeout=ai_timeout,
        input_mode=input_mode,
        human_agent=human_agent,
        questions=questions,
        consent_form_path=consent_form_path,
    )
    app.run()
