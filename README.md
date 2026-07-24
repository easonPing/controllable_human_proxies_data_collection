# Moving Out Environment

Moving Out Environment, a physically grounded human-AI collaboration

---

## Installation

Set up the environment using conda and pip:

```bash
conda create -n movingout_test_git python=3.10
conda activate movingout_test_git
git clone git@github.com:live-robotics-uva/moving_out_env.git
cd moving_out_env
pip install -e .
```

---

## Running the Environment

### Option 1: Keyboard Control (for basic testing)

```bash
python ./run_env_keyboard.py --map_name HandOff
```

### Option 2: Joystick Control (recommended)

```bash
python ./run_env_joystick.py --map_name HandOff
```

---

## Control Guide

### Keyboard Controls (Testing Only)

**Agent 1:**

* `WASD`: Move up, left, down, right
* `H` (hold): Move backwards
* `Spacebar`: Pick up or drop item

**Agent 2:**

* `Arrow Keys`: Move up, left, down, right
* `Ctrl` (hold): Move backwards
* `Enter`: Pick up or drop item

### Joystick Controls (Recommended)

* **Left Stick**: Move the agent
* **R Button**: Pick up or drop item
* **ZL (hold)**: Move backwards

> **Note:** Keyboard controls are limited in direction and speed. For a smoother and more complete gameplay experience, joystick control is strongly recommended.

---

## Custom Maps

To add a custom map:

1. Refer to the existing map definitions in `moving_out/maps/maps_v1`.
2. Create a new `.json` file describing your custom map.
3. Add the new map name to `AVAILABLE_MAPS` in `moving_out/env_parameters.py`.

---

## Data Collection and Replay

Start one human + AI collection session:

```bash
python scripts/data_collector.py
```

The Pygame window collects the participant's Computing ID and email, the
human-controlled agent color, the AI model, and the map. By default the human
uses the first connected joystick; pass `--input keyboard` to test with WASD,
`H` (reverse), and Space (pick up/drop). The other agent runs in a child process
and exchanges state/actions with the environment process over a localhost
socket. The AI Model dropdown provides the seeded `random` placeholder and
`diffusion_policy_hand_off`. The diffusion model automatically selects the
robot-1 or robot-2 checkpoint according to the AI-controlled color. Its
checkpoints were trained for `HandOff`; other maps remain selectable but are
not expected to produce reliable behavior.

Useful optional arguments:

```bash
python scripts/data_collector.py --input keyboard
python scripts/data_collector.py --output saved_datasets --ai-seed 123 --ai-timeout 2
```

The diffusion model performs a warm-up before the worker reports ready and
uses a minimum 10-second response timeout even when `--ai-timeout` is lower.
During a game, the environment continues stepping at a target 10 Hz while an
AI plan is pending. Diffusion returns an eight-action plan tagged with the
state's control step. When the plan arrives, the collector maps its first
action to the current control step, executes the first two actions
preferentially, and uses the remaining plan as a fallback. It records the
source step for latency analysis but does not skip predicted actions based on
latency. If no plan action is available, the AI agent stays still. The timeout
is a failure threshold and does not pause the environment loop.

One invocation records one session. A schema-v3 JSON file stores session
metadata, the trajectory, AI-plan provenance, and questionnaire answers. The
questionnaire appears after the map is completed or the environment reaches
its existing 1000-step limit. By default it presents the nine collaboration
statements as one question per page, each answered on a 7-point Likert scale.

Replay either a legacy trajectory or a new versioned session file:

```bash
python scripts/replay_collected_trajectory.py --json_path saved_datasets/HandOff_YYYYMMDDHHMMSS_SESSION.json --fps 10
```

Run the collection tests:

```bash
python -m unittest discover -s tests -v
```



## Acknowledgements

This environment is built upon the foundations of the following projects:

* [MAGICAL](https://github.com/qxcv/magical)
* [X-MAGICAL](https://github.com/kevinzakka/x-magical)

We thank the authors for their contributions.

---

For issues or contributions, feel free to open an issue or pull request on GitHub.
