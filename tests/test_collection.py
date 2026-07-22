import json
import math
import os
import time
import unittest
import uuid

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from moving_out.benchmarks.moving_out import MovingOutEnv
from moving_out.collection.ai_runtime import (
    AIProcessClient,
    JSONMessageBuffer,
    RandomModel,
    encode_message,
    validate_action,
)
from moving_out.collection.app import (
    CollectionApp,
    DEFAULT_QUESTIONS,
    LIKERT_OPTIONS,
    compose_joint_action,
    validate_email,
)
from moving_out.collection.records import (
    CollectionConfig,
    QuestionnaireItem,
    SessionRecord,
    create_session_path,
    save_session,
)
from moving_out.utils.joystick_controller import JoystickController
from scripts.replay_collected_trajectory import extract_replay_data


class CollectionConfigTests(unittest.TestCase):
    def test_agent_mapping_and_joint_action_order(self):
        blue = CollectionConfig("id", "a@b.com", "blue", "random", "HandOff")
        red = CollectionConfig(
            "id", "a@b.com", "red", "random", "HandOff", "keyboard"
        )
        self.assertEqual((blue.human_agent_index, blue.ai_agent_index), (0, 1))
        self.assertEqual((red.human_agent_index, red.ai_agent_index), (1, 0))
        self.assertEqual(red.input_mode, "keyboard")
        human = [0.1, 0.2, False]
        ai = [0.3, -0.4, True]
        self.assertEqual(compose_joint_action(human, ai, "blue"), [human, ai])
        self.assertEqual(compose_joint_action(human, ai, "red"), [ai, human])

    def test_email_validation(self):
        self.assertTrue(validate_email("person@example.com"))
        self.assertFalse(validate_email("person"))
        self.assertFalse(validate_email("person@example"))
        self.assertFalse(validate_email("person @example.com"))


class RecordTests(unittest.TestCase):
    def test_versioned_session_is_saved_atomically(self):
        question = QuestionnaireItem("test_question", ["a", "b"], answer="a")
        record = SessionRecord(
            metadata={"computing_id": "abc", "email": "a@b.com"},
            trajectory=[{"id": "HandOff", "action": [[0, 0, False]] * 2}],
            questionnaire=[question],
        )
        directory = os.path.join(
            os.getcwd(), "tests", f".tmp-record-{uuid.uuid4().hex}"
        )
        os.makedirs(directory)
        try:
            path = create_session_path(directory, "HandOff", "12345678-rest")
            save_session(path, record)
            with open(path, "r", encoding="utf-8") as stream:
                saved = json.load(stream)
            self.assertEqual(saved["schema_version"], 2)
            self.assertEqual(saved["metadata"]["computing_id"], "abc")
            self.assertEqual(saved["questionnaire"][0]["answer"], "a")
            self.assertNotIn("abc", os.path.basename(path))
        finally:
            if os.path.exists(path):
                os.unlink(path)
            os.rmdir(directory)

    def test_question_rejects_answer_outside_options(self):
        with self.assertRaises(ValueError):
            QuestionnaireItem("q", ["a", "b"], answer="c")


class CollectionUITests(unittest.TestCase):
    def test_config_screen_uses_white_background_and_twelve_map_images(self):
        app = CollectionApp(output_dir=os.path.join(os.getcwd(), "saved_datasets"))
        try:
            app._draw_config()
            self.assertEqual(
                tuple(app.screen.get_at((0, 0))[:3]), (255, 255, 255)
            )
            self.assertEqual(len(app.map_selector.options), 12)
            self.assertEqual(app.map_selector.selected, "HandOff")
            self.assertEqual(app.model_selector.options, ["random"])
            self.assertEqual(
                pygame.display.get_caption()[0], "Moving Out Data Collection"
            )
            app.computing_id_input.text = "abc123"
            app.email_input.text = "person@example.com"
            app.agent_selector.selected = "blue"
            self.assertEqual(app._config_errors(), {})
        finally:
            pygame.key.stop_text_input()

    def test_completed_questionnaire_saves_and_exits_without_saved_screen(self):
        app = CollectionApp(input_mode="keyboard")
        try:
            app.record = SessionRecord(metadata={})
            app._save_record = lambda: None
            app._submit_questionnaire()
            self.assertEqual(app.record.metadata["session_status"], "complete")
            self.assertEqual(app.state, "COMPLETE")
            self.assertFalse(app.running)
        finally:
            pygame.key.stop_text_input()

    def test_game_view_matches_legacy_frame_plus_overlay_dimensions(self):
        app = CollectionApp(input_mode="keyboard")
        try:
            app.frame = pygame.surfarray.array3d(
                pygame.Surface((512, 512))
            ).swapaxes(0, 1)
            app.frame[:, :] = [10, 20, 30]
            app._set_game_display()
            app._draw_game(waiting=True)
            self.assertEqual(app.screen.get_size(), (512, 612))
            self.assertEqual(tuple(app.screen.get_at((250, 250))[:3]), (10, 20, 30))
            self.assertEqual(tuple(app.screen.get_at((20, 562))[:3]), (50, 50, 50))
        finally:
            app._set_main_display("Moving Out Data Collection")
            pygame.key.stop_text_input()

    def test_default_questionnaire_is_nine_single_page_likert_questions(self):
        self.assertEqual(len(DEFAULT_QUESTIONS), 9)
        self.assertTrue(DEFAULT_QUESTIONS[0].question.startswith("Teamwork:"))
        self.assertEqual(DEFAULT_QUESTIONS[0].options, LIKERT_OPTIONS)

        app = CollectionApp(input_mode="keyboard")
        try:
            app.record = SessionRecord(
                metadata={},
                questionnaire=[
                    QuestionnaireItem(item.question, list(item.options))
                    for item in DEFAULT_QUESTIONS[:2]
                ],
            )
            app.survey_index = 0
            option, option_rect = app._survey_option_rects()[0]
            choose = pygame.event.Event(
                pygame.MOUSEBUTTONDOWN, button=1, pos=option_rect.center
            )
            app._handle_survey_events([choose])
            self.assertEqual(app.record.questionnaire[0].answer, option)
            next_event = pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=pygame.Rect(350, 690, 200, 56).center,
            )
            app._handle_survey_events([next_event])
            self.assertEqual(app.survey_index, 1)
        finally:
            pygame.key.stop_text_input()

    def test_ai_model_dropdown_and_keyboard_mode(self):
        app = CollectionApp(
            output_dir=os.path.join(os.getcwd(), "saved_datasets"),
            input_mode="keyboard",
        )
        try:
            open_event = pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=app.model_selector.rect.center,
            )
            self.assertTrue(app.model_selector.handle_event(open_event))
            self.assertTrue(app.model_selector.expanded)
            _, option_rect = app.model_selector._option_rects()[0]
            choose_event = pygame.event.Event(
                pygame.MOUSEBUTTONDOWN, button=1, pos=option_rect.center
            )
            app.model_selector.handle_event(choose_event)
            self.assertEqual(app.model_selector.selected, "random")
            self.assertFalse(app.model_selector.expanded)

            app.computing_id_input.text = "abc123"
            app.email_input.text = "person@example.com"
            app.agent_selector.selected = "blue"
            started = []
            app._start_session = lambda: started.append(True)
            start_event = pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=app.start_button.rect.center,
            )
            app._handle_config_events([start_event])
            self.assertEqual(started, [True])

            app.controller = object()
            key_event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w)
            app._handle_controller_events([key_event])
            self.assertTrue(app._input_has_started())

            class FakeKeyboardController:
                def get_actions(self):
                    return [1, 0.5, 1], [0, 0, 0]

            app.controller = FakeKeyboardController()
            self.assertEqual(app._get_human_action(), [1.0, 0.5, True])
        finally:
            pygame.key.stop_text_input()


class AIRuntimeTests(unittest.TestCase):
    def test_random_model_is_seeded_and_valid(self):
        first = RandomModel()
        second = RandomModel()
        first.reset(agent_index=1, seed=42)
        second.reset(agent_index=1, seed=42)
        first_action = validate_action(first.predict({}))
        second_action = validate_action(second.predict({}))
        self.assertEqual(first_action, second_action)
        self.assertGreaterEqual(first_action[0], 0.0)
        self.assertLessEqual(first_action[0], 1.0)
        self.assertGreaterEqual(first_action[1], -math.pi)
        self.assertLessEqual(first_action[1], math.pi)

    def test_message_buffer_handles_fragmented_and_joined_frames(self):
        payload = encode_message({"type": "one", "value": 1}) + encode_message(
            {"type": "two", "value": 2}
        )
        reader = JSONMessageBuffer()
        messages = []
        for start, end in ((0, 2), (2, 9), (9, len(payload))):
            messages.extend(reader.feed(payload[start:end]))
        self.assertEqual([message["type"] for message in messages], ["one", "two"])

    def test_worker_process_round_trip(self):
        client = AIProcessClient(response_timeout=2.0)
        try:
            client.start("random", agent_index=1, seed=123, startup_timeout=5.0)
            client.request_action(7, {"robot_1": {}, "robot_2": {}})
            deadline = time.monotonic() + 3.0
            action = None
            while action is None and time.monotonic() < deadline:
                action = client.poll_action()
                time.sleep(0.01)
            self.assertIsNotNone(action)
            validate_action(action)
        finally:
            client.close()


class JoystickControllerTests(unittest.TestCase):
    def make_controller_without_hardware(self):
        controller = JoystickController.__new__(JoystickController)
        controller.reset()
        controller.last_angle = [0, 0]
        controller.trigger_threshold = 0.3
        controller.reverse_action = False
        return controller

    def test_single_joystick_action_does_not_need_second_device(self):
        controller = self.make_controller_without_hardware()
        controller.axis[0][0] = 1.0
        action = controller.get_action(0, poll_events=False)
        self.assertEqual(len(action), 3)
        self.assertGreater(action[0], 0)
        self.assertEqual(controller.axis[1], [0, 0, 0, 0])

    def test_meaningful_event_starts_only_its_joystick(self):
        controller = self.make_controller_without_hardware()
        event = pygame.event.Event(
            pygame.JOYAXISMOTION, joy=0, axis=0, value=0.8
        )
        controller.process_events([event])
        self.assertEqual(controller.start_moving, [True, False])

    def test_zero_axis_resets_motion_and_quick_button_tap_is_latched(self):
        controller = self.make_controller_without_hardware()
        move = pygame.event.Event(
            pygame.JOYAXISMOTION, joy=0, axis=0, value=0.8
        )
        center = pygame.event.Event(
            pygame.JOYAXISMOTION, joy=0, axis=0, value=0.0
        )
        press = pygame.event.Event(
            pygame.JOYBUTTONDOWN, joy=0, button=5
        )
        release = pygame.event.Event(
            pygame.JOYBUTTONUP, joy=0, button=5
        )
        controller.process_events([move, center, press, release])
        action = controller.get_action(0, poll_events=False)
        self.assertEqual(controller.axis[0][1], 0)
        self.assertTrue(action[2])
        self.assertFalse(controller.get_action(0, poll_events=False)[2])


class EnvironmentAndReplayTests(unittest.TestCase):
    def test_full_environment_state_round_trips_to_ai_and_drives_one_step(self):
        env = MovingOutEnv(
            use_state=False, map_name="Simple", max_episode_steps=1000
        )
        client = AIProcessClient(response_timeout=2.0)
        try:
            client.start("random", agent_index=1, seed=9, startup_timeout=5.0)
            client.request_action(1, env.get_all_states())
            deadline = time.monotonic() + 3.0
            ai_action = None
            while ai_action is None and time.monotonic() < deadline:
                ai_action = client.poll_action()
                time.sleep(0.01)
            self.assertIsNotNone(ai_action)
            joint_action = compose_joint_action(
                [0.0, 0.0, False], ai_action, "blue"
            )
            env.step(joint_action)
            self.assertEqual(env._episode_steps, 2)
        finally:
            client.close()
            env.close()

    def test_current_two_internal_steps_per_joint_step_is_preserved(self):
        env = MovingOutEnv(
            use_state=False, map_name="Simple", max_episode_steps=4
        )
        action = [[0.0, 0.0, False], [0.0, 0.0, False]]
        try:
            _, _, _, first_truncated, _ = env.step(action)
            _, _, _, second_truncated, _ = env.step(action)
            self.assertFalse(first_truncated)
            self.assertTrue(second_truncated)
            self.assertEqual(env._episode_steps, 4)
        finally:
            env.close()

    def test_replay_loader_accepts_legacy_and_v2(self):
        step = {"id": "HandOff", "action": [[0, 0, False], [0, 0, False]]}
        self.assertEqual(extract_replay_data([step]), ("HandOff", [step]))
        self.assertEqual(extract_replay_data([[step]]), ("HandOff", [step]))
        v2 = {
            "schema_version": 2,
            "metadata": {"map_id": "PassOrSplit"},
            "trajectory": [step],
        }
        self.assertEqual(extract_replay_data(v2), ("PassOrSplit", [step]))


if __name__ == "__main__":
    unittest.main()
