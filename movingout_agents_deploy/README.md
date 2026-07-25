# MovingOut Agents — Deployment Package

Pretrained agents for the MovingOut two-robot cooperative environment.
Inference is pure PyTorch; the only other requirement is the `moving_out`
environment package to play in.

## Contents

```
movingout_agent.py        agent class (torch only)
run_agent.py              runnable entry point
checkpoints/
  <Map>-<Method>-blue.pt  policy for robot 1 (blue)
  <Map>-<Method>-red.pt   policy for robot 2 (red)
```

Maps: `EfficientRoutes`, `SequentialRotations`.
Methods: `GAMMA`, `COMEDI`, `PRISM`.

## Setup

```bash
pip install torch
# plus the MovingOut environment:
git clone git@github.com:live-robotics-uva/moving_out_env.git
pip install -e moving_out_env
```

## Run

```bash
python run_agent.py --map EfficientRoutes --method PRISM --episodes 3
python run_agent.py --map SequentialRotations --method GAMMA --mode det --render
```

Prints per-episode step count and score; `--render` also saves
`episode_<n>.gif`. `--mode` selects stochastic (default) or deterministic
actions.aw

## Using an agent in your own loop

```python
from moving_out.benchmarks.moving_out import MovingOutEnv
from movingout_agents_deploy import MovingOutAgent

blue = MovingOutAgent.load("checkpoints/EfficientRoutes-PRISM-blue.pt")
env = MovingOutEnv(map_name="EfficientRoutes", reward_setting="dense")
env.reset()
obs = env.get_state()                      # [obs_robot1, obs_robot2]
raw = blue.act(obs[0], deterministic=False)
action_robot1 = blue.env_action(raw, holding=env.if_hold[0])
# ... build robot 2's action the same way (or from a human), then:
# env.step([action_robot1, action_robot2])
```

Interface details:

- Each robot is driven independently from its own 121-dim observation
  (`env.get_state()`); blue = robot 1, red = robot 2.
- `agent.act(...)` returns a raw action in [-1,1]^3:
  `[forward, turn, want_hold]`, where `want_hold > 0` means the robot wants
  to be holding an item.
- `agent.env_action(raw, holding)` converts a raw action into what
  `MovingOutEnv.step` expects (turn scaled by pi, hold intent turned into
  the grab-button press). Always pass the robot's current `env.if_hold[i]`.

## Human-AI play

Control one robot from your input device and query the agent only for the
other robot: replace one of the two actions in the loop above with the
human's action. Agents are fully decentralized — they never need the other
player's observation.

## Data collector integration

The repository data collector exposes each method in stochastic and
deterministic modes. `EfficientRoutes` and `SequentialRotations` select their
matching checkpoints. Other maps use `EfficientRoutes` weights and record that
fallback in the saved session metadata.
