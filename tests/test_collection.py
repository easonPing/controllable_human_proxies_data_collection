import json
import math
import os
import time
import unittest
import uuid

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

from moving_out.benchmarks.moving_out import MovingOutEnv
from moving_out.collection.ai_runtime import (
    ActionPlan,
    AIProcessClient,
    DiffusionPolicyHandOffModel,
    JSONMessageBuffer,
    MODEL_REGISTRY,
    PlanResult,
    RandomModel,
    encode_message,
    validate_action,
    validate_action_plan,
)
from moving_out.collection.app import (
    CollectionApp,
    DEFAULT_QUESTIONS,
    IDLE_AI_ACTION,
    LIKERT_OPTIONS,
    align_action_plan,
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
            self.assertEqual(saved["schema_version"], 3)
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
            self.assertEqual(
                app.model_selector.options,
                ["random", "diffusion_policy_hand_off"],
            )
            self.assertEqual(
                pygame.display.get_caption()[0], "Moving Out Data Collection"
            )
            app.computing_id_input.text = "abc123"
            app.email_input.text = "person@example.com"
            app.agent_selector.selected = "blue"
            self.assertEqual(app._config_errors(), {})

            app.model_selector.selected = "diffusion_policy_hand_off"
            app.map_selector.selected = "Simple"
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

    def test_game_steps_and_records_current_state_while_ai_is_pending(self):
        from unittest.mock import patch

        class PendingAIClient:
            pending = True

            def poll_plan(self):
                return None

            def close(self):
                pass

        class FakeController:
            def __init__(self):
                self.calls = 0

            def get_actions(self):
                self.calls += 1
                return [0.5, 0.25, False], [0.0, 0.0, False]

        class FakeEnv:
            def __init__(self):
                self.steps = 0
                self._episode_steps = 0

            def get_all_states(self):
                return {"state_step": self.steps}

            def step(self, action):
                self.steps += 1
                self._episode_steps += 2
                return None, [0.0, 0.0], False, False, {"eval_score": 0.0}

            def render(self, mode):
                self.mode = mode
                return np.zeros((8, 8, 3), dtype=np.uint8)

            def close(self):
                pass

        app = CollectionApp(input_mode="keyboard")
        try:
            app.config = CollectionConfig(
                "id", "person@example.com", "blue", "random", "HandOff",
                "keyboard",
            )
            app.record = SessionRecord(metadata={})
            app.env = FakeEnv()
            app.controller = FakeController()
            app.ai_client = PendingAIClient()
            app.next_control_at = 0.0

            monotonic_values = [
                0.0,
                0.0,
                0.05,
                0.1,
                0.1,
                0.2,
                0.2,
                0.31,
                0.31,
            ]
            with patch(
                "moving_out.collection.app.time.monotonic",
                side_effect=monotonic_values,
            ):
                for _ in range(5):
                    app._update_game()

            self.assertEqual(app.joint_steps, 4)
            self.assertEqual(app.env.steps, 4)
            self.assertEqual(app.controller.calls, 4)
            self.assertEqual(len(app.record.trajectory), 4)
            step = app.record.trajectory[0]
            self.assertEqual(step["control_step"], 0)
            self.assertEqual(step["states"], {"state_step": 0})
            self.assertEqual(step["ai_action_source"], "idle")
            self.assertIsNone(step["ai_plan_source_step"])
            self.assertEqual(step["action"][1], IDLE_AI_ACTION)
            self.assertEqual(
                [entry["states"]["state_step"] for entry in app.record.trajectory],
                [0, 1, 2, 3],
            )
        finally:
            app._close_ai()
            app._close_environment()
            pygame.key.stop_text_input()

    def test_new_plan_immediately_replaces_an_old_tail(self):
        app = CollectionApp(input_mode="keyboard")
        try:
            app.config = CollectionConfig(
                "id",
                "person@example.com",
                "blue",
                "diffusion_policy_hand_off",
                "HandOff",
                "keyboard",
            )
            app.joint_steps = 103
            app.active_action_buffer = [
                {
                    "action": [0.1, 0.0, False],
                    "request_id": 1,
                    "source_step": 90,
                    "plan_index": 7,
                    "target_step": 103,
                    "usage": "tail",
                }
            ]
            actions = [[index / 10, 0.0, False] for index in range(8)]
            app._install_plan(
                PlanResult(
                    request_id=2,
                    source_step=103,
                    actions=actions,
                    preferred_action_count=2,
                )
            )
            self.assertEqual(
                [entry["request_id"] for entry in app.active_action_buffer],
                [2] * 8,
            )
            self.assertEqual(app.active_action_buffer[0]["plan_index"], 0)
            self.assertEqual(app.active_action_buffer[0]["usage"], "preferred")
        finally:
            pygame.key.stop_text_input()


class AIRuntimeTests(unittest.TestCase):
    def test_diffusion_model_is_registered_with_expected_runtime_settings(self):
        self.assertEqual(
            list(MODEL_REGISTRY),
            ["random", "diffusion_policy_hand_off"],
        )
        self.assertEqual(DiffusionPolicyHandOffModel.startup_timeout, 30.0)
        self.assertEqual(
            DiffusionPolicyHandOffModel.minimum_response_timeout, 10.0
        )
        self.assertEqual(RandomModel.startup_timeout, 5.0)
        self.assertEqual(RandomModel.minimum_response_timeout, 0.0)
        self.assertEqual(
            DiffusionPolicyHandOffModel.checkpoint_path(0).name,
            "dp_handoff_robot1.pt",
        )
        self.assertEqual(
            DiffusionPolicyHandOffModel.checkpoint_path(1).name,
            "dp_handoff_robot2.pt",
        )
        with self.assertRaises(ValueError):
            DiffusionPolicyHandOffModel.checkpoint_path(2)

    def test_diffusion_model_encodes_and_selects_its_agent_observation(self):
        class FakeEncoder:
            def get_state_by_current_obs_states(self, state):
                self.state = state
                return [[0.0] * 121, [1.0] * 121]

        class FakePolicy:
            def get_action_plan(self, observation):
                self.observation = observation
                return [[-0.25, 0.5, True]] * 8

        model = DiffusionPolicyHandOffModel()
        model.agent_index = 1
        model._encoder = FakeEncoder()
        model._policy = FakePolicy()

        state = {"robot_1": {}, "robot_2": {}, "items": []}
        plan = validate_action_plan(model.predict_plan(state))
        self.assertEqual(model._encoder.state, state)
        self.assertEqual(model._policy.observation, [1.0] * 121)
        self.assertEqual(len(plan.actions), 8)
        self.assertEqual(plan.preferred_action_count, 2)
        self.assertEqual(plan.actions[0], [-0.25, 0.5, True])

    def test_diffusion_model_rejects_wrong_observation_size(self):
        class FakeEncoder:
            def get_state_by_current_obs_states(self, state):
                return [[0.0] * 120, [0.0] * 120]

        model = DiffusionPolicyHandOffModel()
        model.agent_index = 0
        model._encoder = FakeEncoder()
        model._policy = object()
        with self.assertRaisesRegex(ValueError, "121-dimensional"):
            model.predict_plan({})

    def test_random_model_is_seeded_and_valid(self):
        first = RandomModel()
        second = RandomModel()
        first.reset(agent_index=1, seed=42)
        second.reset(agent_index=1, seed=42)
        first_plan = validate_action_plan(first.predict_plan({}))
        second_plan = validate_action_plan(second.predict_plan({}))
        first_action = first_plan.actions[0]
        second_action = second_plan.actions[0]
        self.assertEqual(first_plan.preferred_action_count, 1)
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

    def test_plan_message_uses_source_step_without_source_time(self):
        payload = encode_message(
            {
                "type": "plan",
                "request_id": 1,
                "source_step": 3,
                "state": {},
            }
        )
        self.assertIn(b"source_step", payload)
        self.assertNotIn(b"source_time", payload)

    def test_action_plan_validator_rejects_bad_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "preferred_action_count"):
            validate_action_plan(
                ActionPlan(
                    actions=[[0.0, 0.0, False]],
                    preferred_action_count=2,
                )
            )

    def test_diffusion_policy_keeps_single_action_api_compatibility(self):
        from movingout_dp_deploy.diffusion_policy import DiffusionPolicy

        policy = DiffusionPolicy.__new__(DiffusionPolicy)
        policy.previous_steps = 0
        policy.pred_horizon = 8
        policy.selected_actions = 2
        policy._history = []
        policy._plan = []
        actions = [[index / 10, 0.0, False] for index in range(8)]
        policy._plan_actions = lambda state: [list(action) for action in actions]

        self.assertEqual(policy.get_action_plan([0.0]), actions)
        self.assertEqual(policy.get_action([0.0]), actions[0])
        self.assertEqual(policy.get_action([0.0]), actions[1])
        self.assertEqual(policy.get_action([0.0]), actions[0])

    def test_plan_alignment_prefers_two_then_keeps_tail(self):
        actions = [[index / 10, 0.0, False] for index in range(8)]
        result = PlanResult(
            request_id=9,
            source_step=100,
            actions=actions,
            preferred_action_count=2,
        )
        aligned = align_action_plan(result, current_step=103)
        self.assertEqual([entry["plan_index"] for entry in aligned], [3, 4, 5, 6, 7])
        self.assertEqual(
            [entry["usage"] for entry in aligned],
            ["preferred", "preferred", "tail", "tail", "tail"],
        )
        self.assertEqual(
            [entry["target_step"] for entry in aligned],
            [103, 104, 105, 106, 107],
        )

    def test_fully_stale_plan_rebases_its_last_two_actions(self):
        actions = [[index / 10, 0.0, False] for index in range(8)]
        result = PlanResult(
            request_id=10,
            source_step=100,
            actions=actions,
            preferred_action_count=2,
        )
        aligned = align_action_plan(result, current_step=112)
        self.assertEqual([entry["plan_index"] for entry in aligned], [6, 7])
        self.assertEqual(
            [entry["target_step"] for entry in aligned], [112, 113]
        )
        self.assertEqual(
            [entry["usage"] for entry in aligned],
            ["stale_rebased", "stale_rebased"],
        )
        with self.assertRaisesRegex(ValueError, "source_step"):
            align_action_plan(result, current_step=99)

    def test_worker_process_round_trip(self):
        client = AIProcessClient(response_timeout=2.0)
        try:
            client.start("random", agent_index=1, seed=123, startup_timeout=5.0)
            client.request_plan(
                7, 42, {"robot_1": {}, "robot_2": {}}
            )
            deadline = time.monotonic() + 3.0
            result = None
            while result is None and time.monotonic() < deadline:
                result = client.poll_plan()
                time.sleep(0.01)
            self.assertIsNotNone(result)
            self.assertEqual(result.request_id, 7)
            self.assertEqual(result.source_step, 42)
            self.assertEqual(result.preferred_action_count, 1)
            validate_action(result.actions[0])
        finally:
            client.close()

    def test_worker_reports_diffusion_startup_error_immediately(self):
        client = AIProcessClient(response_timeout=2.0)
        try:
            with self.assertRaisesRegex(RuntimeError, "agent_index"):
                client.start(
                    "diffusion_policy_hand_off",
                    agent_index=2,
                    seed=123,
                    startup_timeout=5.0,
                )
        finally:
            client.close()


@unittest.skipUnless(
    os.environ.get("RUN_DIFFUSION_SMOKE") == "1",
    "Set RUN_DIFFUSION_SMOKE=1 to load both real checkpoints.",
)
class DiffusionCheckpointSmokeTests(unittest.TestCase):
    def test_collection_app_starts_with_diffusion_runtime_limits(self):
        app = CollectionApp(input_mode="keyboard", ai_timeout=2.0)
        try:
            app.computing_id_input.text = "smoke-test"
            app.email_input.text = "smoke@example.com"
            app.agent_selector.selected = "blue"
            app.model_selector.selected = "diffusion_policy_hand_off"
            app.map_selector.selected = "HandOff"
            app._start_session()

            self.assertEqual(app.state, "WAITING_FOR_INPUT")
            self.assertIsNotNone(app.ai_client)
            self.assertEqual(app.ai_client.response_timeout, 10.0)
            self.assertEqual(app.config.ai_agent_index, 1)
            app._start_game_after_input()
            app._update_game()
            self.assertEqual(app.joint_steps, 1)
            self.assertEqual(app.record.trajectory[0]["control_step"], 0)
            self.assertEqual(
                app.record.trajectory[0]["ai_action_source"], "idle"
            )
            self.assertTrue(app.ai_client.pending)

            deadline = time.monotonic() + 8.0
            while (
                all(
                    step["ai_action_source"] == "idle"
                    for step in app.record.trajectory
                )
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
                app._update_game()
            model_steps = [
                step
                for step in app.record.trajectory
                if step["ai_action_source"] != "idle"
            ]
            self.assertTrue(model_steps)
            self.assertIsNotNone(model_steps[0]["ai_plan_source_step"])
            self.assertGreater(app.joint_steps, 1)
        finally:
            app._close_ai()
            app._close_environment()
            app._set_main_display("Moving Out Data Collection")
            pygame.key.stop_text_input()

    def test_both_real_checkpoints_respond_over_socket(self):
        env = MovingOutEnv(use_state=False, map_name="HandOff")
        try:
            state = env.get_all_states()
            for agent_index in (0, 1):
                with self.subTest(agent_index=agent_index):
                    client = AIProcessClient(response_timeout=10.0)
                    try:
                        client.start(
                            "diffusion_policy_hand_off",
                            agent_index=agent_index,
                            seed=123,
                            startup_timeout=30.0,
                        )
                        client.request_plan(agent_index + 1, 0, state)
                        deadline = time.monotonic() + 12.0
                        result = None
                        while result is None and time.monotonic() < deadline:
                            result = client.poll_plan()
                            time.sleep(0.01)
                        self.assertIsNotNone(result)
                        self.assertEqual(len(result.actions), 8)
                        self.assertEqual(result.preferred_action_count, 2)
                        for action in result.actions:
                            validate_action(action)
                    finally:
                        client.close()
        finally:
            env.close()


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
            client.request_plan(1, 0, env.get_all_states())
            deadline = time.monotonic() + 3.0
            result = None
            while result is None and time.monotonic() < deadline:
                result = client.poll_plan()
                time.sleep(0.01)
            self.assertIsNotNone(result)
            joint_action = compose_joint_action(
                [0.0, 0.0, False], result.actions[0], "blue"
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

    def test_replay_loader_accepts_legacy_v2_and_v3(self):
        step = {"id": "HandOff", "action": [[0, 0, False], [0, 0, False]]}
        self.assertEqual(extract_replay_data([step]), ("HandOff", [step]))
        self.assertEqual(extract_replay_data([[step]]), ("HandOff", [step]))
        v2 = {
            "schema_version": 2,
            "metadata": {"map_id": "PassOrSplit"},
            "trajectory": [step],
        }
        self.assertEqual(extract_replay_data(v2), ("PassOrSplit", [step]))
        v3 = {
            "schema_version": 3,
            "metadata": {"map_id": "HandOff"},
            "trajectory": [step],
        }
        self.assertEqual(extract_replay_data(v3), ("HandOff", [step]))


if __name__ == "__main__":
    unittest.main()
