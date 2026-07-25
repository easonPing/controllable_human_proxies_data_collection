import json
import os
import sys
import time
import unittest
import uuid
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

from moving_out.collection.app import CollectionApp
from moving_out.collection.records import (
    QuestionnaireItem,
    StudyEpisode,
    StudyQuestionnaire,
    StudyQuestionnaireItem,
    StudyRecord,
    create_study_path,
    save_study,
)
from moving_out.collection.study_app import (
    FORMAL_JOINT_STEP_LIMIT,
    STUDY_MODEL_REGISTRY,
    STUDY_SEQUENCE,
    TUTORIAL_JOINT_STEP_LIMIT,
    StudyCollectionApp,
)
from scripts.replay_collected_trajectory import extract_replay_data


class StudyRecordTests(unittest.TestCase):
    def test_schema_v4_stores_independent_episodes_and_questionnaires(self):
        record = StudyRecord(
            metadata={"session_id": "study-id"},
            tutorial={"status": "completed"},
            episodes=[
                StudyEpisode(
                    metadata={
                        "episode_index": 1,
                        "map_id": "EfficientRoutes",
                    },
                    trajectory=[{"id": "EfficientRoutes", "action": []}],
                )
            ],
            questionnaires=[
                StudyQuestionnaire(
                    metadata={
                        "questionnaire_index": 1,
                        "condition": "stochastic",
                    },
                    items=[
                        StudyQuestionnaireItem(
                            "Question",
                            ["1", "2"],
                            skipped=True,
                        )
                    ],
                )
            ],
        )
        directory = os.path.join(
            os.getcwd(),
            "tests",
            f".tmp-study-record-{uuid.uuid4().hex}",
        )
        os.makedirs(directory)
        try:
            path = create_study_path(directory, "12345678-rest")
            save_study(path, record)
            with open(path, "r", encoding="utf-8") as stream:
                saved = json.load(stream)
        finally:
            if os.path.exists(path):
                os.unlink(path)
            os.rmdir(directory)

        self.assertEqual(saved["schema_version"], 4)
        self.assertEqual(
            saved["episodes"][0]["trajectory"][0]["id"],
            "EfficientRoutes",
        )
        self.assertTrue(
            saved["questionnaires"][0]["items"][0]["skipped"]
        )


class DataCollectorCLITests(unittest.TestCase):
    def test_bare_command_uses_fixed_study_schedule_and_blue_default(self):
        from scripts import data_collector

        with (
            patch.object(sys, "argv", ["data_collector.py"]),
            patch.object(
                data_collector, "run_study_collection_app"
            ) as run_study,
            patch.object(data_collector, "run_collection_app") as run_test,
        ):
            data_collector.main()

        run_test.assert_not_called()
        run_study.assert_called_once_with(
            output_dir="saved_datasets",
            ai_seed=None,
            ai_timeout=2.0,
            input_mode="joystick",
            human_agent="blue",
        )

    def test_formal_arguments_and_test_models_routing(self):
        from scripts import data_collector

        with (
            patch.object(
                sys,
                "argv",
                [
                    "data_collector.py",
                    "--agent",
                    "red",
                    "--input",
                    "keyboard",
                ],
            ),
            patch.object(
                data_collector, "run_study_collection_app"
            ) as run_study,
        ):
            data_collector.main()
        self.assertEqual(run_study.call_args.kwargs["human_agent"], "red")
        self.assertEqual(run_study.call_args.kwargs["input_mode"], "keyboard")

        with (
            patch.object(
                sys,
                "argv",
                ["data_collector.py", "--test-models", "--input", "keyboard"],
            ),
            patch.object(data_collector, "run_collection_app") as run_test,
        ):
            data_collector.main()
        self.assertEqual(run_test.call_args.kwargs["input_mode"], "keyboard")

    def test_model_argument_is_removed(self):
        from scripts import data_collector

        with (
            patch.object(
                sys,
                "argv",
                ["data_collector.py", "--model", "gamma"],
            ),
            self.assertRaises(SystemExit),
        ):
            data_collector.main()

    def test_test_models_rejects_cli_agent(self):
        from scripts import data_collector

        with (
            patch.object(
                sys,
                "argv",
                ["data_collector.py", "--test-models", "--agent", "red"],
            ),
            self.assertRaises(SystemExit),
        ):
            data_collector.main()


class StudyCollectionUITests(unittest.TestCase):
    def make_app(self, **kwargs):
        return StudyCollectionApp(input_mode="keyboard", **kwargs)

    def tearDown(self):
        pygame.key.stop_text_input()

    def test_formal_config_only_requires_identity(self):
        app = self.make_app()
        try:
            app.computing_id_input.text = "abc123"
            app.email_input.text = "person@example.com"
            app.agent_selector.selected = None
            self.assertEqual(app._config_errors(), {})
            app._draw_config()
            self.assertEqual(
                tuple(app.screen.get_at((0, 0))[:3]),
                (255, 255, 255),
            )
        finally:
            app._close_ai()
            app._close_environment()

    def test_consent_parses_docx_and_left_stick_scrolls(self):
        app = self.make_app()
        try:
            self.assertEqual(
                app.consent_form_path.name,
                "#6932 IRB-SBS Consent Form.docx",
            )
            paragraphs = app._read_consent_document()
            app._layout_consent_document(paragraphs)
            self.assertGreater(len(app.consent_lines), 50)
            self.assertGreater(app._consent_max_scroll, 0)
            for removed_widget in (
                "print_name_input",
                "date_input",
                "signature_input",
            ):
                self.assertFalse(hasattr(app, removed_widget))
            self.assertFalse(
                any(
                    line.text.startswith("Print Name:")
                    for line in app.consent_lines
                )
            )

            event = pygame.event.Event(
                pygame.JOYAXISMOTION,
                joy=0,
                axis=1,
                value=1.0,
            )
            app._handle_consent_events([event])
            app._update_consent_scroll()
            self.assertGreater(app.consent_scroll, 0)
            previous = app.consent_scroll
            event = pygame.event.Event(
                pygame.JOYAXISMOTION,
                joy=0,
                axis=1,
                value=-1.0,
            )
            app._handle_consent_events([event])
            app._update_consent_scroll()
            self.assertLess(app.consent_scroll, previous)
        finally:
            app._close_ai()
            app._close_environment()

    def test_agree_saves_separate_consent_and_study_json(self):
        directory = os.path.join(
            os.getcwd(),
            "tests",
            f".tmp-study-consent-{uuid.uuid4().hex}",
        )
        os.makedirs(directory)
        app = self.make_app(output_dir=directory, ai_seed=42)
        try:
            paragraphs = app._read_consent_document()
            app._layout_consent_document(paragraphs)
            app.consent_scroll = app._consent_max_scroll
            app._position_consent_widgets()
            self.assertTrue(app._consent_can_agree())
            self.assertTrue(
                app.consent_viewport.contains(app.agree_button.rect)
            )
            app._draw_consent()

            prepared = []
            app._prepare_tutorial = lambda: prepared.append(True)
            app._agree_to_consent()
            self.assertEqual(prepared, [True])
            self.assertIsNotNone(app.consent_path)
            self.assertIsNotNone(app.study_path)
            with open(app.consent_path, "r", encoding="utf-8") as stream:
                consent = json.load(stream)
            with open(app.study_path, "r", encoding="utf-8") as stream:
                study = json.load(stream)
            self.assertEqual(consent["schema_version"], 2)
            self.assertEqual(study["schema_version"], 4)
            self.assertEqual(
                consent["consent_id"],
                study["metadata"]["consent_id"],
            )
            for removed_field in ("print_name", "date", "signature"):
                self.assertNotIn(removed_field, consent)
                self.assertNotIn(removed_field, study["metadata"])
            self.assertEqual(
                consent["document_sha256"],
                study["metadata"]["consent_document_sha256"],
            )
        finally:
            app._close_ai()
            app._close_environment()
            for path in (app.consent_path, app.study_path):
                if path and os.path.exists(path):
                    os.unlink(path)
            consent_directory = os.path.join(directory, "consents")
            if os.path.isdir(consent_directory):
                os.rmdir(consent_directory)
            os.rmdir(directory)

    def test_tutorial_stage_detection_and_skip_inputs(self):
        app = self.make_app()
        try:
            app.study_human_agent = "blue"
            app.tutorial_start_position = (0.0, 0.0)
            before = {
                "robot_1": {
                    "pos": [0.0, 0.0],
                    "holded_item_id": None,
                }
            }
            moved = {
                "robot_1": {
                    "pos": [0.1, 0.0],
                    "holded_item_id": None,
                }
            }
            app._update_tutorial_stage(before, moved, [1.0, 0.0, False])
            self.assertEqual(app.tutorial_stage, 1)
            holding = {
                "robot_1": {
                    "pos": [0.1, 0.0],
                    "holded_item_id": 0,
                }
            }
            app._update_tutorial_stage(
                moved,
                holding,
                [0.0, 0.0, True],
            )
            self.assertEqual(app.tutorial_stage, 2)
            backed = {
                "robot_1": {
                    "pos": [0.08, 0.0],
                    "holded_item_id": 0,
                }
            }
            app._update_tutorial_stage(
                holding,
                backed,
                [-1.0, 0.0, False],
            )
            self.assertEqual(app.tutorial_stage, 3)
            self.assertTrue(
                app._tutorial_skip_requested(
                    [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_x)]
                )
            )
            self.assertTrue(
                app._tutorial_skip_requested(
                    [
                        pygame.event.Event(
                            pygame.JOYBUTTONDOWN,
                            joy=0,
                            button=3,
                        )
                    ]
                )
            )
        finally:
            app._close_ai()
            app._close_environment()

    def test_tutorial_uses_joint_step_limit_without_recording_trajectory(self):
        class FakeEnv:
            def __init__(self):
                self._episode_steps = 0
                self.position = 0.0

            def get_all_states(self):
                return {
                    "robot_1": {
                        "pos": [self.position, 0.0],
                        "holded_item_id": None,
                    },
                    "robot_2": {
                        "pos": [0.0, 0.0],
                        "holded_item_id": None,
                    },
                }

            def step(self, action):
                self.position += 0.01
                self._episode_steps += 2
                return None, None, False, False, {"eval_score": 0.0}

            def render(self, mode):
                return np.zeros((8, 8, 3), dtype=np.uint8)

            def close(self):
                pass

        class FakeController:
            def get_actions(self):
                return [1.0, 0.0, False], [0.0, 0.0, False]

        app = self.make_app()
        try:
            app.env = FakeEnv()
            app.controller = FakeController()
            app.config = type(
                "Config",
                (),
                {"human_agent": "blue"},
            )()
            app.tutorial_start_position = (0.0, 0.0)
            app.tutorial_joint_steps = TUTORIAL_JOINT_STEP_LIMIT - 1
            app.tutorial_next_control_at = 0.0
            reasons = []
            app._finish_tutorial = reasons.append
            with patch(
                "moving_out.collection.study_app.time.monotonic",
                return_value=1.0,
            ):
                app._update_tutorial()
            self.assertEqual(reasons, ["step_limit"])
            self.assertIsNone(app.study_record)
        finally:
            app._close_ai()
            app._close_environment()

    def test_tutorial_overlay_uses_elapsed_monotonic_time(self):
        app = self.make_app()
        try:
            app.frame = np.zeros((8, 8, 3), dtype=np.uint8)
            app.tutorial_started_monotonic = 100.0
            with (
                patch(
                    "moving_out.collection.study_app.time.monotonic",
                    return_value=102.5,
                ),
                patch.object(
                    app.overlay_renderer,
                    "display_image_with_overlay",
                    return_value=np.zeros((16, 8, 3), dtype=np.uint8),
                ) as render_overlay,
            ):
                app._draw_tutorial()
            self.assertEqual(render_overlay.call_args.args[0].shape, (8, 8, 3))
            self.assertEqual(render_overlay.call_args.args[1], 0.0)
            self.assertEqual(render_overlay.call_args.args[2], 2.5)
        finally:
            app._close_ai()
            app._close_environment()

    def test_fixed_episode_order_and_model_resolution(self):
        self.assertEqual(
            STUDY_SEQUENCE,
            (
                ("gamma", "stochastic", "EfficientRoutes"),
                ("gamma", "stochastic", "SequentialRotations"),
                ("comedi", "stochastic", "EfficientRoutes"),
                ("comedi", "stochastic", "SequentialRotations"),
                ("prism", "stochastic", "EfficientRoutes"),
                ("prism", "stochastic", "SequentialRotations"),
            ),
        )
        self.assertEqual(FORMAL_JOINT_STEP_LIMIT, 500)
        for family in ("gamma", "comedi", "prism"):
            self.assertEqual(
                StudyCollectionApp._formal_model_name(
                    family,
                    "stochastic",
                ),
                STUDY_MODEL_REGISTRY[family]["stochastic"],
            )
            self.assertEqual(
                list(STUDY_MODEL_REGISTRY[family]),
                ["stochastic"],
            )

    def test_formal_limit_uses_joint_control_steps(self):
        app = self.make_app()
        try:
            app.state = "GAME"
            app.joint_steps = FORMAL_JOINT_STEP_LIMIT - 1
            app.current_episode = StudyEpisode(metadata={})
            reasons = []
            app._finish_episode = reasons.append

            def advance_one_joint_step():
                app.joint_steps += 1

            with patch.object(
                CollectionApp,
                "_update_game",
                side_effect=advance_one_joint_step,
            ):
                app._update_game()
            self.assertEqual(reasons, ["step_limit"])
        finally:
            app._close_ai()
            app._close_environment()

    def test_episode_transition_inserts_questionnaire_after_episode_two(self):
        app = self.make_app()
        try:
            app.study_record = StudyRecord(metadata={})
            app.study_path = "unused"
            app._save_study = lambda: None
            app._close_ai = lambda: None
            app._close_environment = lambda: None
            app.current_episode = StudyEpisode(
                metadata={
                    "status": "running",
                    "model_family": "gamma",
                }
            )
            app.current_episode_index = 0
            app.joint_steps = 5
            app.score = 0.5
            prepared = []
            surveys = []
            app._prepare_formal_episode = prepared.append
            app._begin_questionnaire = (
                lambda model_family, number: surveys.append(
                    (model_family, number)
                )
            )
            app._finish_episode("completed")
            self.assertEqual(prepared, [1])
            self.assertEqual(surveys, [])

            app.current_episode = StudyEpisode(
                metadata={
                    "status": "running",
                    "model_family": "gamma",
                }
            )
            app.current_episode_index = 1
            app._finish_episode("step_limit")
            self.assertEqual(surveys, [("gamma", 1)])
        finally:
            app._close_ai()
            app._close_environment()

    def test_questionnaire_skip_is_model_specific(self):
        app = self.make_app()
        try:
            app.study_record = StudyRecord(metadata={})
            app.study_path = "unused"
            app._save_study = lambda: None
            app.question_templates = [
                QuestionnaireItem("Q1", ["1", "2"]),
                QuestionnaireItem("Q2", ["1", "2"]),
            ]
            app._begin_questionnaire("gamma", 1)
            skip_event = pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=pygame.Rect(585, 690, 150, 56).center,
            )
            app._handle_survey_events([skip_event])
            first = app.study_record.questionnaires[0]
            self.assertTrue(first.items[0].skipped)
            self.assertIsNone(first.items[0].answer)
            self.assertEqual(app.survey_index, 1)

            app._begin_questionnaire("comedi", 2)
            second = app.study_record.questionnaires[1]
            self.assertEqual(second.metadata["model_family"], "comedi")
            self.assertEqual(second.metadata["condition"], "stochastic")
            self.assertFalse(second.items[0].skipped)
            self.assertIsNone(second.items[0].answer)
        finally:
            app._close_ai()
            app._close_environment()


class StudyReplayTests(unittest.TestCase):
    def test_schema_v4_selects_requested_episode(self):
        first = {
            "episode_index": 1,
            "map_id": "EfficientRoutes",
            "trajectory": [
                {"id": "EfficientRoutes", "action": [[0, 0, False]] * 2}
            ],
        }
        second = {
            "episode_index": 2,
            "map_id": "SequentialRotations",
            "trajectory": [
                {
                    "id": "SequentialRotations",
                    "action": [[0, 0, False]] * 2,
                }
            ],
        }
        payload = {
            "schema_version": 4,
            "metadata": {},
            "episodes": [first, second],
        }
        map_id, steps = extract_replay_data(payload, episode=2)
        self.assertEqual(map_id, "SequentialRotations")
        self.assertIs(steps, second["trajectory"])
        with self.assertRaisesRegex(ValueError, "between 1 and 2"):
            extract_replay_data(payload, episode=3)


@unittest.skipUnless(
    os.environ.get("RUN_STUDY_SMOKE") == "1",
    "Set RUN_STUDY_SMOKE=1 to run the real six-episode study workers.",
)
class StudyRuntimeSmokeTests(unittest.TestCase):
    def test_real_workers_follow_the_six_episode_schedule(self):
        app = StudyCollectionApp(
            input_mode="keyboard",
            human_agent="blue",
            ai_timeout=2.0,
        )
        try:
            app.computing_id_input.text = "study-smoke"
            app.email_input.text = "smoke@example.com"
            app.study_ai_seed = 100
            app.study_record = StudyRecord(
                metadata={
                    "started_at": None,
                    "session_status": "formal_pending",
                }
            )
            app._save_study = lambda: None

            for index, (
                model_family,
                condition,
                map_id,
            ) in enumerate(STUDY_SEQUENCE):
                with self.subTest(
                    index=index,
                    model=model_family,
                    condition=condition,
                    map=map_id,
                ):
                    app._prepare_formal_episode(index)
                    self.assertEqual(app.state, "WAITING_FOR_INPUT")
                    self.assertEqual(app.config.map_id, map_id)
                    self.assertEqual(
                        app.config.ai_model,
                        f"movingout_{model_family}_{condition}",
                    )
                    self.assertEqual(
                        app.current_episode.metadata["ai_checkpoint_name"],
                        f"{map_id}-{model_family.upper()}-red.pt",
                    )
                    app._start_game_after_input()
                    deadline = time.monotonic() + 3.0
                    model_steps = []
                    while time.monotonic() < deadline and not model_steps:
                        app._update_game()
                        model_steps = [
                            step
                            for step in app.current_episode.trajectory
                            if step["ai_action_source"] != "idle"
                        ]
                        time.sleep(0.01)
                    self.assertTrue(model_steps)
                    self.assertEqual(model_steps[0]["ai_plan_index"], 0)
                    app._close_ai()
                    app._close_environment()
        finally:
            app._close_ai()
            app._close_environment()
            pygame.key.stop_text_input()

    def test_real_one_step_end_to_end_orchestration(self):
        app = StudyCollectionApp(
            input_mode="keyboard",
            human_agent="red",
            ai_timeout=2.0,
        )
        try:
            app.computing_id_input.text = "study-flow-smoke"
            app.email_input.text = "smoke@example.com"
            app.study_ai_seed = 200
            app.study_record = StudyRecord(
                metadata={
                    "started_at": None,
                    "session_status": "formal_pending",
                }
            )
            app._save_study = lambda: None
            with patch(
                "moving_out.collection.study_app.FORMAL_JOINT_STEP_LIMIT",
                1,
            ):
                app._prepare_formal_episode(0)
                for episode_index in range(6):
                    self.assertEqual(app.state, "WAITING_FOR_INPUT")
                    self.assertEqual(
                        app.current_episode_index,
                        episode_index,
                    )
                    app._start_game_after_input()
                    deadline = time.monotonic() + 3.0
                    while app.state == "GAME" and time.monotonic() < deadline:
                        app._update_game()
                        time.sleep(0.01)
                    self.assertNotEqual(app.state, "GAME")

                    if episode_index in {1, 3, 5}:
                        self.assertEqual(app.state, "SURVEY")
                        while app.state == "SURVEY":
                            index = app.survey_index
                            app.record.questionnaire[index].answer = "1"
                            app.current_questionnaire.items[index].answer = "1"
                            app._advance_questionnaire()

            self.assertEqual(app.state, "COMPLETE")
            self.assertEqual(len(app.study_record.episodes), 6)
            self.assertEqual(
                [len(episode.trajectory) for episode in app.study_record.episodes],
                [1, 1, 1, 1, 1, 1],
            )
            self.assertEqual(len(app.study_record.questionnaires), 3)
            self.assertEqual(
                [
                    questionnaire.metadata["model_family"]
                    for questionnaire in app.study_record.questionnaires
                ],
                ["gamma", "comedi", "prism"],
            )
            self.assertEqual(
                app.study_record.metadata["session_status"],
                "complete",
            )
        finally:
            app._close_ai()
            app._close_environment()
            pygame.key.stop_text_input()


if __name__ == "__main__":
    unittest.main()
