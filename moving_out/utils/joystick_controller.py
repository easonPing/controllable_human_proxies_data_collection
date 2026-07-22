"""A user interface for teleoperating an agent in an x-magical environment.

Modified from https://github.com/unixpickle/obs-tower2/blob/master/obs_tower2/recorder/env_interactor.py
"""


import time
from typing import List
import numpy as np
import pygame

from .utils import resultant_vector


class JoystickController:
    """User interface for interacting in an x-magical environment."""

    def __init__(
        self,
        resolution: int = 384,
        initial_angle=None,
    ):
        del resolution
        self.reset()
        self.last_angle = list(initial_angle or [0, 0])

        self.trigger_threshold = 0.3
        self.reverse_action = False
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            print("No joystick detected")
            # sys.exit()
        self.joysticks = []
        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            self.joysticks.append(joystick)
            print(f"Detected {i}: {joystick.get_name()}")

    def update_last_angle(self, angle, robot_id):
        self.last_angle[robot_id] = angle

    def _get_action_for_joystick(self, joystick_index: int):
        """Translate one physical joystick state to one environment action."""

        if joystick_index not in (0, 1):
            raise ValueError("Only joystick indices 0 and 1 are supported.")

        x_displacement = self.axis[joystick_index][0]
        y_displacement = self.axis[joystick_index][1]

        if x_displacement != 0 or y_displacement != 0:
            if (
                np.linalg.norm((x_displacement, y_displacement), ord=2)
                <= self.trigger_threshold
            ):
                magnitude = 0
                angle = self.last_angle[joystick_index]

            else:
                magnitude, angle = resultant_vector(
                    [[0, 0], [x_displacement, 0]], [[0, 0], [0, y_displacement]]
                )
                magnitude = (magnitude - self.trigger_threshold) / (
                    1 - self.trigger_threshold
                )
                magnitude = 1 if magnitude >= 1 else magnitude
                magnitude = -1 if magnitude <= -1 else magnitude
                if self.move_back[joystick_index]:
                    magnitude = -magnitude
                    angle = (angle + np.pi) % (2 * np.pi)
                if self.only_rotate[joystick_index]:
                    magnitude = 0

        else:
            magnitude = 0
            angle = self.last_angle[joystick_index]

        self.update_last_angle(angle, joystick_index)
        return [magnitude, angle, self.hold[joystick_index]]

    def get_action(self, joystick_index: int = 0, poll_events: bool = True):
        """Return one joystick action for human+AI collection.

        The collection app passes its already-consumed Pygame event list to
        :meth:`process_events`, then calls this method with ``poll_events=False``.
        Existing scripts can keep the default and let the controller poll.
        """

        if poll_events:
            self.joystick_event_process()
        action = self._get_action_for_joystick(joystick_index)
        self.hold[joystick_index] = False
        return action

    def get_actions(self) -> List[float]:
        self.joystick_event_process()
        physical_actions = [
            self._get_action_for_joystick(0),
            self._get_action_for_joystick(1),
        ]
        if self.reverse_action:
            action = [physical_actions[1], physical_actions[0]]
        else:
            action = physical_actions

        self.hold = [False, False]
        return action

    def reset(self):
        self._started = False
        self._finish_early = False
        self._last_image = None
        self.move_back = [False, False]
        self.only_rotate = [False, False]
        self.hold = [False, False]
        self.axis = [[0, 0, 0, 0], [0, 0, 0, 0]]
        self.start_moving = [False, False]

        self.action_hold_button = [5, 10]
        self.action_only_rotate_button = [4]

    def process_events(self, events):
        """Update controller state from events owned by an outer Pygame loop."""

        joystick_offset = [0.00, 0.9]
        for event in events:
            joystick_index = getattr(event, "joy", None)
            if joystick_index not in (0, 1):
                if event.type == pygame.QUIT:
                    self._finish_early = True
                continue
            if event.type == pygame.QUIT:
                self._finish_early = True
            elif event.type == pygame.JOYBUTTONDOWN:
                self.start_moving[joystick_index] = True
                if event.button in self.action_hold_button:
                    self.hold[joystick_index] = True
                if event.button in self.action_only_rotate_button:
                    self.only_rotate[joystick_index] = True
            elif event.type == pygame.JOYBUTTONUP:
                # ``hold`` is a one-step interaction pulse. Keep a button-down
                # latched until get_action() consumes it so a quick tap is not
                # lost between the 60 Hz UI loop and the 10 Hz environment loop.
                if event.button in self.action_only_rotate_button:
                    self.only_rotate[joystick_index] = False
            elif event.type == pygame.JOYAXISMOTION:
                if event.axis in (0, 1) and abs(event.value) > self.trigger_threshold:
                    self.start_moving[joystick_index] = True
                if event.axis == 0:
                    if abs(event.value) <= joystick_offset[0]:
                        self.axis[joystick_index][1] = 0
                    elif (
                        abs(event.value) > joystick_offset[0]
                        and abs(event.value) < joystick_offset[1]
                    ):
                        self.axis[joystick_index][1] = -(
                            event.value - joystick_offset[0]
                        ) / (joystick_offset[1] - joystick_offset[0])
                    elif abs(event.value) > joystick_offset[1]:
                        self.axis[joystick_index][1] = (
                            -1 if (event.value > 0) else 1
                        )
                if event.axis == 1:
                    if abs(event.value) <= joystick_offset[0]:
                        self.axis[joystick_index][0] = 0
                    elif (
                        abs(event.value) > joystick_offset[0]
                        and abs(event.value) < joystick_offset[1]
                    ):
                        self.axis[joystick_index][0] = -(
                            event.value - joystick_offset[0]
                        ) / (joystick_offset[1] - joystick_offset[0])
                    elif abs(event.value) > joystick_offset[1]:
                        self.axis[joystick_index][0] = (
                            -1 if (event.value > 0) else 1
                        )
                if event.axis == 2:
                    if abs(event.value) <= joystick_offset[0]:
                        self.axis[joystick_index][2] = 0
                    elif (
                        abs(event.value) > joystick_offset[0]
                        and abs(event.value) < joystick_offset[1]
                    ):
                        self.axis[joystick_index][2] = -(
                            event.value - joystick_offset[0]
                        ) / (joystick_offset[1] - joystick_offset[0])
                    elif abs(event.value) > joystick_offset[1]:
                        self.axis[joystick_index][2] = (
                            -1 if (event.value > 0) else 1
                        )
                if event.axis == 3:
                    if abs(event.value) <= joystick_offset[0]:
                        self.axis[joystick_index][3] = 0
                    elif (
                        abs(event.value) > joystick_offset[0]
                        and abs(event.value) < joystick_offset[1]
                    ):
                        self.axis[joystick_index][3] = -(
                            event.value - joystick_offset[0]
                        ) / (joystick_offset[1] - joystick_offset[0])
                    elif abs(event.value) > joystick_offset[1]:
                        self.axis[joystick_index][3] = (
                            -1 if (event.value > 0) else 1
                        )

                if event.axis == 4:
                    if event.value > 0.5:
                        self.move_back[joystick_index] = True
                    elif event.value < 0.5:
                        self.move_back[joystick_index] = False
            elif event.type == pygame.JOYHATMOTION:
                pass

    def joystick_event_process(self, events=None):
        if events is None:
            events = pygame.event.get()
        self.process_events(events)



