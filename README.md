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

Start the formal six-episode data-collection study:

```bash
python scripts/data_collector.py
python scripts/data_collector.py --agent red
python scripts/data_collector.py --agent blue --input keyboard
```

Formal mode defaults to `--agent blue --input joystick`.
`--agent` is the human-controlled color. The CONFIG page collects only the
Computing ID and email, followed by the scrollable IRB consent form, an
Agree-only consent acknowledgement, and a step-by-step `Simple`-map tutorial. The
tutorial is not recorded. It ends when all objects reach a target, after 3000
joint control steps, or when keyboard/controller X skips it. Consent can be
scrolled with the mouse, keyboard, or the first joystick's left stick. No
printed name, date, or signature is requested; reaching the bottom and clicking
Agree records the acknowledgement.

The six recorded episodes run in this fixed order:

1. GAMMA stochastic `EfficientRoutes`
2. GAMMA stochastic `SequentialRotations`
3. GAMMA questionnaire
4. COMEDI stochastic `EfficientRoutes`
5. COMEDI stochastic `SequentialRotations`
6. COMEDI questionnaire
7. PRISM stochastic `EfficientRoutes`
8. PRISM stochastic `SequentialRotations`
9. PRISM questionnaire

Each formal episode has a 500 joint-control-step limit, corresponding to
50 seconds at the target 10 Hz control rate. The limit remains step-based
rather than wall-clock based. A fresh nine-item questionnaire follows each
model's two episodes. The saved schema-v4 study contains separate episode
trajectories and questionnaires; consent acknowledgements are stored
separately under `saved_datasets/consents/`.

Use the preserved single-map model-testing pipeline when selecting arbitrary
models, agents, and maps:

```bash
python scripts/data_collector.py --test-models
python scripts/data_collector.py --test-models --input keyboard
```

This mode retains the seeded `random` placeholder,
`diffusion_policy_hand_off`, and stochastic/deterministic variants of GAMMA,
COMEDI, and PRISM in the UI. The human uses the first connected joystick by
default; keyboard mode uses WASD, `H` (reverse), and Space (pick up/drop). The
other agent runs in a child process and exchanges state/actions with the
environment process over a localhost socket.

The diffusion checkpoints were trained for `HandOff`; other maps remain
selectable but are not expected to produce reliable behavior. GAMMA, COMEDI,
and PRISM use map-specific weights for `EfficientRoutes` and
`SequentialRotations`. On every other map they remain runnable using the
`EfficientRoutes` weights, and the fallback is recorded in session metadata.

Useful optional arguments:

```bash
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

GAMMA, COMEDI, and PRISM are single-step reactive policies. Their stochastic
variants use `--ai-seed` for reproducible sampling; deterministic variants use
the policy mean. Their desired-hold output is converted to the environment's
grab/release toggle using the AI robot's current holding state.

The `--test-models` pipeline continues to save its existing schema-v3
single-map record. Formal collection saves one schema-v4 study file and updates
it atomically after each stage. Both formats retain AI-plan provenance.

Replay either a legacy trajectory or a new versioned session file:

```bash
python scripts/replay_collected_trajectory.py --json_path saved_datasets/HandOff_YYYYMMDDHHMMSS_SESSION.json --fps 10
python scripts/replay_collected_trajectory.py --json_path saved_datasets/study_YYYYMMDDHHMMSS_SESSION.json --episode 3 --fps 10
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
