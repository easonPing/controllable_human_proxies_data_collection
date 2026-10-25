import argparse
import json
import os
import sys


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

import cv2
from moving_out.benchmarks.moving_out import MovingOutEnv


def load_json(file_path):
    """Load the JSON data from the specified file."""
    with open(file_path, "r") as file:
        return json.load(file)


def extract_replay_data(json_data, episode=1):
    """Return ``(map_id, steps)`` for legacy and versioned recordings."""

    metadata = {}
    if isinstance(json_data, dict):
        schema_version = json_data.get("schema_version")
        if schema_version == 4:
            episodes = json_data.get("episodes")
            if not isinstance(episodes, list) or not episodes:
                raise ValueError("Invalid schema-v4 study: no episodes.")
            if (
                isinstance(episode, bool)
                or not isinstance(episode, int)
                or not 1 <= episode <= len(episodes)
            ):
                raise ValueError(
                    f"--episode must be between 1 and {len(episodes)}"
                )
            selected = episodes[episode - 1]
            if not isinstance(selected, dict):
                raise ValueError("Invalid schema-v4 episode.")
            metadata = selected
            steps = selected.get("trajectory")
        elif schema_version in {2, 3}:
            if episode != 1:
                raise ValueError(
                    "--episode can only be greater than 1 for schema-v4 studies."
                )
            metadata = json_data.get("metadata", {})
            steps = json_data.get("trajectory")
        else:
            raise ValueError("Unsupported trajectory object schema.")
        if not isinstance(metadata, dict) or not isinstance(steps, list):
            raise ValueError(
                f"Invalid schema-v{schema_version} trajectory format."
            )
    elif isinstance(json_data, list):
        if not json_data:
            raise ValueError("Invalid trajectory format: expected a non-empty list.")
        if isinstance(json_data[0], list):
            if not json_data[0]:
                raise ValueError("Invalid trajectory format: empty episode list.")
            steps = json_data[0]
        else:
            steps = json_data
    else:
        raise ValueError("Invalid trajectory format.")

    if not steps:
        raise ValueError("Trajectory contains no recorded steps.")
    first_step = steps[0]
    if not isinstance(first_step, dict) or "action" not in first_step:
        raise ValueError("Invalid trajectory format: missing 'action'.")
    map_id = metadata.get("map_id", first_step.get("id"))
    if not map_id:
        raise ValueError("Invalid trajectory format: missing map id.")
    return map_id, steps


def replay_trajectory(json_path, fps, episode=1):
    json_data = load_json(json_path)
    map_id, steps = extract_replay_data(json_data, episode=episode)
    env = MovingOutEnv(use_state=False, map_name=map_id)

    delay_ms = max(1, int(1000 / fps))
    try:
        for step in steps:
            action = step["action"]
            obs, _, terminated, truncated, info = env.step(action)
            rgb_obs = env.render("rgb_array")
            bgr_obs = cv2.cvtColor(rgb_obs, cv2.COLOR_RGB2BGR)
            cv2.imshow("Replay", bgr_obs)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q"):
                break
            if terminated or truncated or info.get("eval_score", 0) >= 1.0:
                break
            if obs is None:
                break
    finally:
        env.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Replay a collected trajectory.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to JSON.")
    parser.add_argument("--fps", type=int, default=10, help="Replay FPS.")
    parser.add_argument(
        "--episode",
        type=int,
        default=1,
        help="1-based episode number for schema-v4 study files (default: 1).",
    )
    args = parser.parse_args()

    if args.fps < 1:
        raise ValueError("--fps must be >= 1")
    if args.episode < 1:
        raise ValueError("--episode must be >= 1")

    replay_trajectory(args.json_path, args.fps, episode=args.episode)


if __name__ == "__main__":
    main()
