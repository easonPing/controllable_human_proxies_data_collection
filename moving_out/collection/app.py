"""Single-window Pygame application for human+AI trajectory collection."""

from __future__ import annotations

import os
import re
import secrets
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame

from moving_out.benchmarks.moving_out import MovingOutEnv
from moving_out.env_parameters import AVAILABLE_MAPS, get_map_image_path
from moving_out.utils.data_collection import DataCollector
from moving_out.utils.joystick_controller import JoystickController
from moving_out.utils.keyboard_controller import KeyboardController

from .ai_runtime import AIProcessClient, MODEL_REGISTRY, PlanResult
from .records import (
    CollectionConfig,
    QuestionnaireItem,
    SessionRecord,
    create_session_path,
    new_session_id,
    now_iso,
    save_session,
    to_jsonable,
)


WINDOW_WIDTH = 900
WINDOW_HEIGHT = 900
UI_FPS = 60
CONTROL_HZ = 10.0
CONTROL_INTERVAL = 1.0 / CONTROL_HZ
IDLE_AI_ACTION = [0.0, 0.0, False]
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
LIKERT_OPTIONS = ["1", "2", "3", "4", "5", "6", "7"]


DEFAULT_QUESTIONS = [
    QuestionnaireItem(
        question=(
            "Teamwork: The other agent and I worked together towards a goal."
        ),
        options=list(LIKERT_OPTIONS),
    ),
    QuestionnaireItem(
        question="Humanlike: The other agent's actions were human-like.",
        options=list(LIKERT_OPTIONS),
    ),
    QuestionnaireItem(
        question=(
            "Reasonable: The other agent always made reasonable actions "
            "throughout the game."
        ),
        options=list(LIKERT_OPTIONS),
    ),
    QuestionnaireItem(
        question=(
            "Follow: The other agent followed my lead when making decisions."
        ),
        options=list(LIKERT_OPTIONS),
    ),
    QuestionnaireItem(
        question=(
            "Physics: The other agent understands how to work with me when "
            "objects have varying physical characteristics."
        ),
        options=list(LIKERT_OPTIONS),
    ),
    QuestionnaireItem(
        question=(
            "Helpfulness: The other agent understands my intention and "
            "proactively helps me when I need assistance."
        ),
        options=list(LIKERT_OPTIONS),
    ),
    QuestionnaireItem(
        question=(
            "Collision: When our movement paths conflict, the other agent "
            "and I can effectively coordinate to avoid collisions."
        ),
        options=list(LIKERT_OPTIONS),
    ),
    QuestionnaireItem(
        question=(
            "Alignment: When moving large items together, our target "
            "directions remain well-aligned."
        ),
        options=list(LIKERT_OPTIONS),
    ),
    QuestionnaireItem(
        question=(
            "Future: I would like to collaborate with the other agent in "
            "future Moving Out tasks."
        ),
        options=list(LIKERT_OPTIONS),
    ),
]


def validate_email(value: str) -> bool:
    return bool(EMAIL_PATTERN.fullmatch(value.strip()))


def compose_joint_action(
    human_action: Sequence[Any], ai_action: Sequence[Any], human_agent: str
) -> List[List[Any]]:
    """Order controls using the environment's blue=0, red=1 convention."""

    human = list(human_action)
    ai = list(ai_action)
    if human_agent == "blue":
        return [human, ai]
    if human_agent == "red":
        return [ai, human]
    raise ValueError(f"Unknown human agent: {human_agent!r}")


def align_action_plan(
    result: PlanResult, current_step: int
) -> List[Dict[str, Any]]:
    """Rebase a returned plan's first action to the next control step."""

    if result.source_step > current_step:
        raise ValueError(
            "AI plan source_step cannot be greater than the current control step."
        )

    aligned_actions: List[Dict[str, Any]] = []
    for plan_index, action in enumerate(result.actions):
        aligned_actions.append(
            {
                "action": list(action),
                "request_id": result.request_id,
                "source_step": result.source_step,
                "plan_index": plan_index,
                "target_step": current_step + plan_index,
                "usage": (
                    "preferred"
                    if plan_index < result.preferred_action_count
                    else "tail"
                ),
            }
        )
    return aligned_actions


class Button:
    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        font: pygame.font.Font,
        base_color: Tuple[int, int, int] = (65, 65, 85),
        hover_color: Tuple[int, int, int] = (95, 95, 125),
    ) -> None:
        self.rect = rect
        self.text = text
        self.font = font
        self.base_color = base_color
        self.hover_color = hover_color

    def is_clicked(self, event: pygame.event.Event, enabled: bool = True) -> bool:
        return bool(
            enabled
            and event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        )

    def draw(
        self,
        screen: pygame.Surface,
        enabled: bool = True,
        selected: bool = False,
        selected_color: Tuple[int, int, int] = (55, 145, 80),
    ) -> None:
        if not enabled:
            color = (45, 45, 50)
            text_color = (125, 125, 130)
        elif selected:
            color = selected_color
            text_color = (255, 255, 255)
        elif self.rect.collidepoint(pygame.mouse.get_pos()):
            color = self.hover_color
            text_color = (255, 255, 255)
        else:
            color = self.base_color
            text_color = (255, 255, 255)
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, (145, 145, 155), self.rect, 1, border_radius=8)
        text = self.font.render(self.text, True, text_color)
        screen.blit(text, text.get_rect(center=self.rect.center))


class TextInput:
    def __init__(
        self,
        rect: pygame.Rect,
        font: pygame.font.Font,
        placeholder: str,
        max_length: int = 128,
    ) -> None:
        self.rect = rect
        self.font = font
        self.placeholder = placeholder
        self.max_length = max_length
        self.text = ""
        self.active = False

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.active = self.rect.collidepoint(event.pos)
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_DELETE:
                self.text = ""
        elif event.type == pygame.TEXTINPUT and self.active:
            printable = "".join(ch for ch in event.text if ch.isprintable())
            room = self.max_length - len(self.text)
            if room > 0:
                self.text += printable[:room]

    def draw(self, screen: pygame.Surface, invalid: bool = False) -> None:
        if invalid:
            border = (220, 70, 70)
        elif self.active:
            border = (105, 170, 245)
        else:
            border = (125, 125, 135)
        pygame.draw.rect(screen, (248, 248, 250), self.rect, border_radius=6)
        pygame.draw.rect(screen, border, self.rect, 2, border_radius=6)
        display_text = self.text or self.placeholder
        color = (25, 25, 30) if self.text else (115, 115, 120)
        rendered = self.font.render(display_text, True, color)
        clip = self.rect.inflate(-18, -8)
        previous_clip = screen.get_clip()
        screen.set_clip(clip)
        screen.blit(rendered, (self.rect.x + 10, self.rect.y + 9))
        if self.active and int(time.monotonic() * 2) % 2 == 0:
            cursor_x = self.rect.x + 10 + self.font.size(self.text)[0]
            pygame.draw.line(
                screen,
                (25, 25, 30),
                (cursor_x, self.rect.y + 9),
                (cursor_x, self.rect.bottom - 9),
                2,
            )
        screen.set_clip(previous_clip)


class ChoiceSelector:
    def __init__(
        self,
        x: int,
        y: int,
        options: Sequence[str],
        font: pygame.font.Font,
        selected: Optional[str] = None,
        width: int = 170,
    ) -> None:
        self.options = list(options)
        self.selected = selected
        self.buttons = [
            Button(pygame.Rect(x + index * (width + 15), y, width, 46), option, font)
            for index, option in enumerate(self.options)
        ]

    def handle_event(self, event: pygame.event.Event) -> None:
        for button in self.buttons:
            if button.is_clicked(event):
                self.selected = button.text

    def draw(
        self,
        screen: pygame.Surface,
        colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ) -> None:
        colors = colors or {}
        for button in self.buttons:
            button.draw(
                screen,
                selected=button.text == self.selected,
                selected_color=colors.get(button.text, (55, 145, 80)),
            )


class DropdownSelector:
    """Compact single-choice dropdown used by registry-backed options."""

    def __init__(
        self,
        rect: pygame.Rect,
        options: Sequence[str],
        font: pygame.font.Font,
    ) -> None:
        if not options:
            raise ValueError("A dropdown requires at least one option.")
        self.rect = rect
        self.options = list(options)
        self.font = font
        self.selected = self.options[0]
        self.expanded = False
        self.option_height = 44

    def _option_rects(self) -> List[Tuple[str, pygame.Rect]]:
        return [
            (
                option,
                pygame.Rect(
                    self.rect.x,
                    self.rect.bottom + index * self.option_height,
                    self.rect.width,
                    self.option_height,
                ),
            )
            for index, option in enumerate(self.options)
        ]

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.expanded:
                self.expanded = False
                return True
            return False
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return self.expanded
        if not self.expanded and self.rect.collidepoint(event.pos):
            self.expanded = True
            return True
        if self.expanded:
            for option, rect in self._option_rects():
                if rect.collidepoint(event.pos):
                    self.selected = option
                    self.expanded = False
                    return True
            self.expanded = False
            return True
        return False

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(screen, (248, 248, 250), self.rect, border_radius=8)
        pygame.draw.rect(screen, (130, 130, 140), self.rect, 2, border_radius=8)
        label = self.font.render(self.selected, True, (25, 25, 30))
        screen.blit(label, (self.rect.x + 16, self.rect.y + 10))
        arrow_x = self.rect.right - 22
        arrow_y = self.rect.centery
        if self.expanded:
            arrow = [
                (arrow_x, arrow_y - 5),
                (arrow_x - 6, arrow_y + 4),
                (arrow_x + 6, arrow_y + 4),
            ]
        else:
            arrow = [
                (arrow_x, arrow_y + 5),
                (arrow_x - 6, arrow_y - 4),
                (arrow_x + 6, arrow_y - 4),
            ]
        pygame.draw.polygon(screen, (45, 45, 50), arrow)

        if self.expanded:
            mouse = pygame.mouse.get_pos()
            for option, rect in self._option_rects():
                if rect.collidepoint(mouse):
                    color = (225, 232, 245)
                elif option == self.selected:
                    color = (215, 238, 220)
                else:
                    color = (248, 248, 250)
                pygame.draw.rect(screen, color, rect, border_radius=5)
                pygame.draw.rect(
                    screen, (130, 130, 145), rect, 1, border_radius=5
                )
                option_text = self.font.render(option, True, (25, 25, 30))
                screen.blit(
                    option_text,
                    (
                        rect.x + 16,
                        rect.y + (rect.height - option_text.get_height()) // 2,
                    ),
                )


class MapSelector:
    """Selected-map preview with a full-window thumbnail picker."""

    def __init__(
        self,
        rect: pygame.Rect,
        options: Sequence[str],
        font: pygame.font.Font,
        small_font: pygame.font.Font,
    ) -> None:
        if not options:
            raise ValueError("At least one map with a preview image is required.")
        self.rect = rect
        self.options = list(options)
        self.font = font
        self.small_font = small_font
        self.selected = "HandOff" if "HandOff" in self.options else self.options[0]
        self.expanded = False
        self.images: Dict[str, pygame.Surface] = {}
        self.thumbnail_images: Dict[str, pygame.Surface] = {}
        for option in self.options:
            image = pygame.image.load(get_map_image_path(option)).convert_alpha()
            self.images[option] = pygame.transform.smoothscale(image, (130, 110))
            self.thumbnail_images[option] = pygame.transform.smoothscale(
                image, (135, 105)
            )

    def _option_rects(self) -> List[Tuple[str, pygame.Rect]]:
        columns = 4
        cell_width = 190
        cell_height = 160
        origin_x = (WINDOW_WIDTH - columns * cell_width) // 2
        origin_y = 155
        return [
            (
                option,
                pygame.Rect(
                    origin_x + (index % columns) * cell_width,
                    origin_y + (index // columns) * cell_height,
                    cell_width - 12,
                    cell_height - 12,
                ),
            )
            for index, option in enumerate(self.options)
        ]

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle an event and report whether the overlay consumed it."""

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.expanded = False
            return True
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return self.expanded
        if not self.expanded and self.rect.collidepoint(event.pos):
            self.expanded = True
            return True
        if self.expanded:
            for option, rect in self._option_rects():
                if rect.collidepoint(event.pos):
                    self.selected = option
                    self.expanded = False
                    return True
            return True
        return False

    def draw_header(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(screen, (248, 248, 250), self.rect, border_radius=8)
        pygame.draw.rect(screen, (130, 130, 140), self.rect, 2, border_radius=8)
        preview = self.images[self.selected]
        screen.blit(preview, (self.rect.x + 12, self.rect.y + 10))
        name = self.font.render(self.selected, True, (25, 25, 30))
        screen.blit(name, (self.rect.x + 165, self.rect.y + 48))
        hint = self.small_font.render("Click to choose a map", True, (105, 105, 115))
        screen.blit(hint, (self.rect.x + 165, self.rect.y + 82))
        arrow = [
            (self.rect.right - 28, self.rect.centery - 5),
            (self.rect.right - 16, self.rect.centery - 5),
            (self.rect.right - 22, self.rect.centery + 4),
        ]
        pygame.draw.polygon(screen, (45, 45, 50), arrow)

    def draw_overlay(
        self, screen: pygame.Surface, title_font: pygame.font.Font
    ) -> None:
        screen.fill((255, 255, 255))
        title = title_font.render("Choose Map", True, (20, 20, 25))
        screen.blit(title, title.get_rect(center=(WINDOW_WIDTH // 2, 70)))
        hint = self.small_font.render(
            "Select a thumbnail (Esc returns)", True, (95, 95, 105)
        )
        screen.blit(hint, hint.get_rect(center=(WINDOW_WIDTH // 2, 112)))
        mouse = pygame.mouse.get_pos()
        for option, rect in self._option_rects():
            if rect.collidepoint(mouse):
                color = (225, 232, 245)
            elif option == self.selected:
                color = (215, 238, 220)
            else:
                color = (248, 248, 250)
            pygame.draw.rect(screen, color, rect, border_radius=8)
            pygame.draw.rect(screen, (125, 125, 140), rect, 1, border_radius=8)
            image = self.thumbnail_images[option]
            image_rect = image.get_rect(midtop=(rect.centerx, rect.y + 8))
            screen.blit(image, image_rect)
            label = self.small_font.render(option, True, (25, 25, 30))
            screen.blit(label, label.get_rect(center=(rect.centerx, rect.bottom - 20)))


class CollectionApp:
    """Own the UI, environment, socket client, and one collection session."""

    def __init__(
        self,
        output_dir: str = "saved_datasets",
        ai_seed: Optional[int] = None,
        ai_timeout: float = 2.0,
        input_mode: str = "joystick",
        questions: Optional[Sequence[QuestionnaireItem]] = None,
    ) -> None:
        if input_mode not in {"joystick", "keyboard"}:
            raise ValueError("input_mode must be 'joystick' or 'keyboard'.")
        self.output_dir = os.path.abspath(os.path.expanduser(output_dir))
        self.requested_ai_seed = ai_seed
        self.ai_timeout = ai_timeout
        self.input_mode = input_mode
        self.question_templates = list(questions or DEFAULT_QUESTIONS)

        pygame.init()
        pygame.joystick.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Moving Out Data Collection")
        pygame.key.start_text_input()
        self.clock = pygame.time.Clock()

        self.title_font = pygame.font.Font(None, 48)
        self.heading_font = pygame.font.Font(None, 34)
        self.body_font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.error_font = pygame.font.Font(None, 24)
        self.overlay_renderer = DataCollector()

        self.state = "CONFIG"
        self.running = True
        self.error_message = ""
        self.status_message = ""

        self.computing_id_input = TextInput(
            pygame.Rect(330, 105, 450, 48), self.body_font, "Enter computing ID"
        )
        self.email_input = TextInput(
            pygame.Rect(330, 180, 450, 48), self.body_font, "Enter email"
        )
        self.agent_selector = ChoiceSelector(
            330, 255, ["blue", "red"], self.body_font, selected=None
        )
        self.model_selector = DropdownSelector(
            pygame.Rect(330, 330, 450, 48),
            list(MODEL_REGISTRY.keys()),
            self.body_font,
        )
        map_options = [
            key
            for key in AVAILABLE_MAPS
            if isinstance(key, str) and os.path.exists(get_map_image_path(key))
        ]
        self.map_selector = MapSelector(
            pygame.Rect(330, 405, 450, 132),
            map_options,
            self.body_font,
            self.small_font,
        )
        self.start_button = Button(
            pygame.Rect(330, 585, 240, 58),
            "Start",
            self.heading_font,
            base_color=(35, 135, 65),
            hover_color=(55, 165, 85),
        )
        self.exit_button = Button(
            pygame.Rect(350, 610, 200, 54), "Exit", self.heading_font
        )

        self.config: Optional[CollectionConfig] = None
        self.record: Optional[SessionRecord] = None
        self.session_path: Optional[str] = None
        self.env: Optional[MovingOutEnv] = None
        self.controller: Optional[Any] = None
        self.input_started = False
        self.ai_client: Optional[AIProcessClient] = None
        self.frame: Optional[np.ndarray] = None
        self.score = 0.0
        self.game_started_monotonic: Optional[float] = None
        self.next_control_at = 0.0
        self.active_action_buffer: List[Dict[str, Any]] = []
        self.next_ai_request_id = 1
        self.last_requested_source_step: Optional[int] = None
        self.joint_steps = 0
        self.survey_index = 0

    def _set_main_display(self, caption: str) -> None:
        if self.screen.get_size() != (WINDOW_WIDTH, WINDOW_HEIGHT):
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(caption)

    def _set_game_display(self) -> None:
        if self.frame is None:
            raise RuntimeError("Cannot size the game window before rendering a frame.")
        combined = self.overlay_renderer.display_image_with_overlay(
            self.frame, 0.0, 0.0
        )
        size = (combined.shape[1], combined.shape[0])
        if self.screen.get_size() != size:
            self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("MovingOut")

    def _config_errors(self) -> Dict[str, str]:
        errors: Dict[str, str] = {}
        if not self.computing_id_input.text.strip():
            errors["computing_id"] = "Computing ID is required."
        if not validate_email(self.email_input.text):
            errors["email"] = "Enter a valid email address."
        if self.agent_selector.selected not in {"blue", "red"}:
            errors["agent"] = "Choose the human-controlled agent."
        return errors

    def _handle_config_events(self, events: Sequence[pygame.event.Event]) -> None:
        if self.map_selector.expanded:
            for event in events:
                self.map_selector.handle_event(event)
            return

        for event in events:
            if self.model_selector.expanded:
                self.model_selector.handle_event(event)
                continue
            if self.model_selector.handle_event(event):
                continue
            if self.map_selector.handle_event(event):
                continue
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
            self.agent_selector.handle_event(event)
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
                    self._start_session()
                return

    def _start_session(self) -> None:
        assert self.agent_selector.selected is not None
        assert self.model_selector.selected is not None
        self.config = CollectionConfig(
            computing_id=self.computing_id_input.text.strip(),
            email=self.email_input.text.strip(),
            human_agent=self.agent_selector.selected,
            ai_model=self.model_selector.selected,
            map_id=self.map_selector.selected,
            input_mode=self.input_mode,
        )
        self.state = "AI_STARTING"
        self._draw_starting()
        pygame.display.flip()

        session_id = new_session_id()
        ai_seed = (
            self.requested_ai_seed
            if self.requested_ai_seed is not None
            else secrets.randbelow(2**31)
        )
        try:
            self.env = MovingOutEnv(
                use_state=False,
                map_name=self.config.map_id,
            )
            if self.input_mode == "joystick":
                self.controller = JoystickController()
            else:
                self.controller = KeyboardController()
            model_class = MODEL_REGISTRY[self.config.ai_model]
            response_timeout = max(
                self.ai_timeout, model_class.minimum_response_timeout
            )
            self.ai_client = AIProcessClient(response_timeout=response_timeout)
            self.ai_client.start(
                model_name=self.config.ai_model,
                agent_index=self.config.ai_agent_index,
                seed=ai_seed,
                startup_timeout=model_class.startup_timeout,
            )
            self.frame = self.env.render("rgb_array")
            self._set_game_display()
            self.record = SessionRecord(
                metadata={
                    "session_id": session_id,
                    "computing_id": self.config.computing_id,
                    "email": self.config.email,
                    "map_id": self.config.map_id,
                    "human_agent": self.config.human_agent,
                    "human_agent_index": self.config.human_agent_index,
                    "ai_agent": self.config.ai_agent,
                    "ai_agent_index": self.config.ai_agent_index,
                    "ai_model": self.config.ai_model,
                    "ai_seed": ai_seed,
                    "input_mode": self.config.input_mode,
                    "created_at": now_iso(),
                    "started_at": None,
                    "ended_at": None,
                    "end_reason": None,
                    "session_status": "waiting_for_input",
                    "control_hz": CONTROL_HZ,
                    "max_episode_steps": self.env.max_episode_steps,
                    "async_ai": True,
                    "ai_plan_horizon": model_class.plan_horizon,
                    "ai_preferred_action_count": (
                        model_class.preferred_action_count
                    ),
                    "ai_fallback_policy": "queued_plan_then_idle",
                    "ai_plan_alignment_policy": "rebase_from_first_action",
                },
                questionnaire=[
                    QuestionnaireItem(item.question, list(item.options))
                    for item in self.question_templates
                ],
            )
            self.session_path = create_session_path(
                self.output_dir, self.config.map_id, session_id
            )
            self.active_action_buffer = []
            self.next_ai_request_id = 1
            self.last_requested_source_step = None
            self.joint_steps = 0
            self.score = 0.0
            self.input_started = False
            self.state = "WAITING_FOR_INPUT"
            if self.input_mode == "joystick":
                print("Please move the first joystick to begin.")
            else:
                print("Please press WASD to begin.")
        except Exception as exc:
            self._close_ai()
            self._close_environment()
            self._set_main_display("Moving Out Data Collection")
            self.record = None
            self.session_path = None
            self.state = "CONFIG"
            self.error_message = f"Could not start session: {exc}"

    def _handle_controller_events(
        self, events: Sequence[pygame.event.Event]
    ) -> None:
        if self.controller is None:
            return
        if self.input_mode == "joystick":
            self.controller.process_events(events)
            return
        start_keys = {
            pygame.K_w,
            pygame.K_a,
            pygame.K_s,
            pygame.K_d,
            pygame.K_SPACE,
            pygame.K_h,
        }
        if any(
            event.type == pygame.KEYDOWN and event.key in start_keys
            for event in events
        ):
            self.input_started = True

    def _input_has_started(self) -> bool:
        if self.controller is None:
            return False
        if self.input_mode == "joystick":
            return bool(self.controller.start_moving[0])
        return self.input_started

    def _get_human_action(self) -> List[Any]:
        if self.controller is None:
            raise RuntimeError("Input controller is not initialized.")
        if self.input_mode == "joystick":
            return list(self.controller.get_action(0, poll_events=False))
        keyboard_action, _ = self.controller.get_actions()
        return [
            float(keyboard_action[0]),
            float(keyboard_action[1]),
            bool(keyboard_action[2]),
        ]

    def _start_game_after_input(self) -> None:
        assert self.record is not None
        started = now_iso()
        self.record.metadata["started_at"] = started
        self.record.metadata["session_status"] = "running"
        self.game_started_monotonic = time.monotonic()
        self.next_control_at = self.game_started_monotonic
        self.state = "GAME"

    def _maybe_request_plan(self) -> None:
        assert self.env is not None
        assert self.ai_client is not None

        if self.ai_client.pending:
            return
        if self.last_requested_source_step == self.joint_steps:
            return

        request_id = self.next_ai_request_id
        source_step = self.joint_steps
        state = self.env.get_all_states()
        self.ai_client.request_plan(request_id, source_step, state)
        self.next_ai_request_id += 1
        self.last_requested_source_step = source_step

    def _install_plan(self, result: PlanResult) -> None:
        assert self.config is not None
        model_class = MODEL_REGISTRY[self.config.ai_model]
        if len(result.actions) != model_class.plan_horizon:
            raise ValueError(
                f"AI model {self.config.ai_model!r} returned "
                f"{len(result.actions)} actions; expected "
                f"{model_class.plan_horizon}."
            )
        if (
            result.preferred_action_count
            != model_class.preferred_action_count
        ):
            raise ValueError(
                f"AI model {self.config.ai_model!r} returned preferred_action_count "
                f"{result.preferred_action_count}; expected "
                f"{model_class.preferred_action_count}."
            )
        self.active_action_buffer = align_action_plan(
            result, current_step=self.joint_steps
        )

    def _service_ai(self) -> None:
        assert self.ai_client is not None

        result = self.ai_client.poll_plan()
        if result is not None:
            self._install_plan(result)
        self._maybe_request_plan()

    def _next_ai_action(self) -> Tuple[List[Any], Dict[str, Any]]:
        while (
            self.active_action_buffer
            and self.active_action_buffer[0]["target_step"] < self.joint_steps
        ):
            del self.active_action_buffer[0]

        if (
            self.active_action_buffer
            and self.active_action_buffer[0]["target_step"] == self.joint_steps
        ):
            provenance = self.active_action_buffer.pop(0)
            return list(provenance["action"]), provenance

        return list(IDLE_AI_ACTION), {
            "request_id": None,
            "source_step": None,
            "plan_index": None,
            "target_step": self.joint_steps,
            "usage": "idle",
        }

    def _update_game(self) -> None:
        assert self.env is not None
        assert self.controller is not None
        assert self.ai_client is not None
        assert self.config is not None
        assert self.record is not None

        try:
            self._service_ai()
        except TimeoutError as exc:
            self._abort_session("ai_timeout", str(exc))
            return
        except (ConnectionError, OSError) as exc:
            self._abort_session("ai_disconnect", str(exc))
            return
        except Exception as exc:
            self._abort_session("ai_runtime_error", str(exc))
            return

        now = time.monotonic()
        if now < self.next_control_at:
            return

        try:
            control_step = self.joint_steps
            state_before_step = self.env.get_all_states()
            human_action = self._get_human_action()
            ai_action, provenance = self._next_ai_action()
            joint_action = compose_joint_action(
                human_action,
                ai_action,
                self.config.human_agent,
            )
            _, reward, terminated, truncated, info = self.env.step(joint_action)
            self.frame = self.env.render("rgb_array")
            self.joint_steps += 1
            eval_score = float(info.get("eval_score", 0.0))
            self.score = max(0.0, min(1.0, eval_score))
            done = bool(terminated or truncated or eval_score >= 1.0)
            self.record.trajectory.append(
                {
                    "red_name": "",
                    "blue_name": "",
                    "time": now_iso(),
                    "id": self.config.map_id,
                    "step": self.joint_steps,
                    "control_step": control_step,
                    "environment_step": int(getattr(self.env, "_episode_steps", 0)),
                    "states": to_jsonable(state_before_step),
                    "rew": to_jsonable(reward),
                    "done": done,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "action": to_jsonable(joint_action),
                    "ai_action_source": provenance["usage"],
                    "ai_plan_request_id": provenance["request_id"],
                    "ai_plan_source_step": provenance["source_step"],
                    "ai_plan_index": provenance["plan_index"],
                    "info": to_jsonable(info),
                    "test_mode": False,
                }
            )
            self.next_control_at = max(
                self.next_control_at + CONTROL_INTERVAL, time.monotonic()
            )

            if terminated or eval_score >= 1.0:
                self._finish_episode("completed")
            elif info.get("physics_nan"):
                self._abort_session("environment_error", "Environment physics failed.")
            elif truncated:
                self._finish_episode("time_limit")
        except Exception as exc:
            self._abort_session("runtime_error", str(exc))

    def _finish_episode(self, reason: str) -> None:
        assert self.record is not None
        self.record.metadata.update(
            {
                "ended_at": now_iso(),
                "end_reason": reason,
                "session_status": "awaiting_questionnaire",
                "joint_steps": self.joint_steps,
                "environment_steps": int(getattr(self.env, "_episode_steps", 0)),
            }
        )
        self._close_ai()
        self._set_main_display("Moving Out Questionnaire")
        try:
            self._save_record()
        except Exception as exc:
            self.state = "ERROR"
            self.error_message = f"Could not save trajectory: {exc}"
            return
        if not self.record.questionnaire:
            self._submit_questionnaire()
            return
        self.survey_index = 0
        self.state = "SURVEY"

    def _abort_session(self, reason: str, message: str) -> None:
        self._close_ai()
        self._set_main_display("Moving Out Data Collection")
        save_error = ""
        if self.record is not None:
            self.record.metadata.update(
                {
                    "ended_at": now_iso(),
                    "end_reason": reason,
                    "session_status": "aborted",
                    "error": message,
                    "joint_steps": self.joint_steps,
                    "environment_steps": int(
                        getattr(self.env, "_episode_steps", 0)
                    ),
                }
            )
            try:
                self._save_record()
            except Exception as exc:
                save_error = f" Saving also failed: {exc}"
        self.error_message = message + save_error
        self.state = "ERROR"

    def _survey_option_rects(self) -> List[Tuple[str, pygame.Rect]]:
        assert self.record is not None
        question = self.record.questionnaire[self.survey_index]
        if question.options == LIKERT_OPTIONS:
            return [
                (option, pygame.Rect(90 + index * 105, 405, 80, 58))
                for index, option in enumerate(question.options)
            ]
        return [
            (option, pygame.Rect(210, 370 + index * 65, 480, 50))
            for index, option in enumerate(question.options)
        ]

    def _handle_survey_events(self, events: Sequence[pygame.event.Event]) -> None:
        assert self.record is not None
        question = self.record.questionnaire[self.survey_index]
        next_rect = pygame.Rect(350, 690, 200, 56)
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for option, rect in self._survey_option_rects():
                    if rect.collidepoint(event.pos):
                        question.answer = option
                        break
                if next_rect.collidepoint(event.pos) and question.answer is not None:
                    if self.survey_index + 1 < len(self.record.questionnaire):
                        self.survey_index += 1
                    else:
                        self._submit_questionnaire()

    def _submit_questionnaire(self) -> None:
        assert self.record is not None
        self.record.metadata["session_status"] = "complete"
        self.record.metadata["questionnaire_completed_at"] = now_iso()
        try:
            self._save_record()
        except Exception as exc:
            self.state = "ERROR"
            self.error_message = f"Could not save questionnaire: {exc}"
            return
        self.state = "COMPLETE"
        self.running = False

    def _save_record(self) -> None:
        if self.record is None or self.session_path is None:
            raise RuntimeError("No active session record to save.")
        save_session(self.session_path, self.record)

    def _handle_close(self) -> None:
        if self.state in {"WAITING_FOR_INPUT", "GAME", "AI_STARTING"}:
            if self.record is not None:
                self.record.metadata.update(
                    {
                        "ended_at": now_iso(),
                        "end_reason": "user_closed",
                        "session_status": "aborted",
                        "joint_steps": self.joint_steps,
                        "environment_steps": int(
                            getattr(self.env, "_episode_steps", 0)
                        ),
                    }
                )
                try:
                    self._save_record()
                except Exception:
                    pass
        elif self.state == "SURVEY" and self.record is not None:
            self.record.metadata["session_status"] = "survey_incomplete"
            try:
                self._save_record()
            except Exception:
                pass
        self.running = False

    def _close_ai(self) -> None:
        if self.ai_client is not None:
            self.ai_client.close()
            self.ai_client = None

    def _close_environment(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None

    def _draw_label(self, text: str, y: int) -> None:
        rendered = self.heading_font.render(text, True, (25, 25, 30))
        self.screen.blit(rendered, (110, y))

    def _draw_config(self) -> None:
        self.screen.fill((255, 255, 255))
        title = self.title_font.render("Data Collection", True, (20, 20, 25))
        self.screen.blit(title, title.get_rect(center=(WINDOW_WIDTH // 2, 48)))
        errors = self._config_errors()

        self._draw_label("Computing ID", 112)
        self.computing_id_input.draw(
            self.screen,
            invalid=bool(
                self.computing_id_input.text and errors.get("computing_id")
            ),
        )
        self._draw_label("Email", 188)
        self.email_input.draw(
            self.screen, invalid=bool(self.email_input.text and errors.get("email"))
        )
        self._draw_label("Human Agent", 263)
        self.agent_selector.draw(
            self.screen,
            colors={"blue": (45, 105, 190), "red": (185, 55, 65)},
        )
        self._draw_label("AI Model", 338)
        self._draw_label("Map", 448)
        self.map_selector.draw_header(self.screen)
        self.start_button.draw(self.screen, enabled=not bool(errors))

        if self.error_message:
            error = self.error_font.render(self.error_message, True, (235, 80, 80))
            self.screen.blit(error, error.get_rect(center=(WINDOW_WIDTH // 2, 685)))
        self.model_selector.draw(self.screen)
        if self.map_selector.expanded:
            self.map_selector.draw_overlay(self.screen, self.title_font)

    def _draw_starting(self) -> None:
        self.screen.fill((0, 0, 0))
        title = self.title_font.render(
            "Starting AI and environment...", True, (255, 255, 255)
        )
        self.screen.blit(title, title.get_rect(center=(WINDOW_WIDTH // 2, 410)))
        hint = self.body_font.render("Please wait", True, (155, 155, 165))
        self.screen.blit(hint, hint.get_rect(center=(WINDOW_WIDTH // 2, 465)))

    def _draw_game(self, waiting: bool = False) -> None:
        if self.frame is None:
            self.screen.fill((0, 0, 0))
            return
        elapsed = 0.0
        if not waiting and self.game_started_monotonic is not None:
            elapsed = time.monotonic() - self.game_started_monotonic
        combined = self.overlay_renderer.display_image_with_overlay(
            self.frame,
            self.score,
            elapsed,
        )
        surface = pygame.surfarray.make_surface(combined.swapaxes(0, 1))
        self.screen.blit(surface, (0, 0))

    @staticmethod
    def _wrap_text(
        text: str, font: pygame.font.Font, max_width: int
    ) -> List[str]:
        words = text.split()
        if not words:
            return [""]
        lines: List[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def _draw_survey(self) -> None:
        assert self.record is not None
        self.screen.fill((255, 255, 255))
        heading = self.title_font.render("Questionnaire", True, (20, 20, 20))
        self.screen.blit(heading, heading.get_rect(center=(WINDOW_WIDTH // 2, 65)))
        counter = self.small_font.render(
            f"Question {self.survey_index + 1} of {len(self.record.questionnaire)}",
            True,
            (95, 95, 100),
        )
        self.screen.blit(counter, counter.get_rect(center=(WINDOW_WIDTH // 2, 105)))
        question = self.record.questionnaire[self.survey_index]
        is_likert = question.options == LIKERT_OPTIONS
        instruction_text = (
            "Please rate the statement using the 7-Point Likert Scale."
            if is_likert
            else "Please select one response."
        )
        instruction = self.body_font.render(
            instruction_text, True, (45, 45, 45)
        )
        self.screen.blit(
            instruction,
            instruction.get_rect(center=(WINDOW_WIDTH // 2, 155)),
        )

        prompt_lines = self._wrap_text(
            question.question, self.heading_font, WINDOW_WIDTH - 140
        )
        prompt_y = 230
        for line in prompt_lines:
            prompt = self.heading_font.render(line, True, (20, 20, 20))
            self.screen.blit(
                prompt, prompt.get_rect(center=(WINDOW_WIDTH // 2, prompt_y))
            )
            prompt_y += self.heading_font.get_linesize() + 5

        for option, rect in self._survey_option_rects():
            selected = question.answer == option
            hovered = rect.collidepoint(pygame.mouse.get_pos())
            if selected:
                fill = (45, 105, 190)
                text_color = (255, 255, 255)
            elif hovered:
                fill = (225, 232, 245)
                text_color = (20, 20, 20)
            else:
                fill = (245, 245, 247)
                text_color = (20, 20, 20)
            pygame.draw.rect(self.screen, fill, rect, border_radius=6)
            pygame.draw.rect(
                self.screen, (80, 80, 85), rect, 2, border_radius=6
            )
            option_text = self.body_font.render(option, True, text_color)
            self.screen.blit(option_text, option_text.get_rect(center=rect.center))

        if is_likert:
            disagree = self.small_font.render(
                "1 - Strongly disagree", True, (65, 65, 70)
            )
            neutral = self.small_font.render("4 - Neutral", True, (65, 65, 70))
            agree = self.small_font.render(
                "7 - Strongly agree", True, (65, 65, 70)
            )
            self.screen.blit(disagree, (90, 485))
            self.screen.blit(
                neutral, neutral.get_rect(center=(WINDOW_WIDTH // 2, 496))
            )
            self.screen.blit(agree, (WINDOW_WIDTH - agree.get_width() - 90, 485))

        is_last = self.survey_index + 1 == len(self.record.questionnaire)
        next_button = Button(
            pygame.Rect(350, 690, 200, 56),
            "Submit" if is_last else "Next",
            self.heading_font,
            base_color=(35, 135, 65),
            hover_color=(55, 165, 85),
        )
        next_button.draw(self.screen, enabled=question.answer is not None)

    def _draw_error(self) -> None:
        self.screen.fill((0, 0, 0))
        title = self.title_font.render("Session stopped", True, (235, 90, 90))
        self.screen.blit(title, title.get_rect(center=(WINDOW_WIDTH // 2, 330)))
        message = self.body_font.render(self.error_message, True, (235, 235, 240))
        self.screen.blit(message, message.get_rect(center=(WINDOW_WIDTH // 2, 405)))
        if self.session_path and self.record is not None:
            saved = self.small_font.render(
                f"Partial data: {self.session_path}", True, (160, 160, 170)
            )
            self.screen.blit(saved, saved.get_rect(center=(WINDOW_WIDTH // 2, 455)))
        self.exit_button.draw(self.screen)

    def run(self) -> None:
        try:
            while self.running:
                events = pygame.event.get()
                if any(event.type == pygame.QUIT for event in events):
                    self._handle_close()
                    continue

                if self.state == "CONFIG":
                    self._handle_config_events(events)
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
                elif self.state == "AI_STARTING":
                    self._draw_starting()
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


def run_collection_app(
    output_dir: str = "saved_datasets",
    ai_seed: Optional[int] = None,
    ai_timeout: float = 2.0,
    input_mode: str = "joystick",
    questions: Optional[Sequence[QuestionnaireItem]] = None,
) -> None:
    app = CollectionApp(
        output_dir=output_dir,
        ai_seed=ai_seed,
        ai_timeout=ai_timeout,
        input_mode=input_mode,
        questions=questions,
    )
    app.run()
