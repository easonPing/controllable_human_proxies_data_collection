# MovingOut — Diffusion Policy inference

Inference-only package for the diffusion behavior-cloning agent on the
MovingOut **HandOff** map. Two trained models (one per robot) drive the two
robots; there is **no training code here** — just load and roll out.

## Contents

| file | what it is |
|---|---|
| `diffusion_model.py` | the diffusion network (`ConditionalUnet1D`), pure `torch`/`einops` |
| `diffusion_policy.py` | `DiffusionPolicy`: observation in, action out (loads a checkpoint, DDIM denoising) |
| `rollout.py` | runnable demo — drives the MovingOut env with the two models |
| `dp_handoff_robot1.pt` | trained model for robot 1 (HandOff) |
| `dp_handoff_robot2.pt` | trained model for robot 2 (HandOff) |

`diffusion_model.py` and `diffusion_policy.py` do **not** depend on the
MovingOut environment, so you can embed the policy in your own runner. Only
`rollout.py` needs the environment.

## Install

```bash
# 1. inference deps
pip install torch diffusers einops numpy imageio

# 2. the MovingOut environment (from the env repo)
git clone https://github.com/live-robotics-uva/moving_out_envw
cd moving_out_env && pip install -e . && cd ..
```

Any recent PyTorch works (CPU is fine — a rollout runs in well under a minute
on CPU).

## Run

```bash
python rollout.py                       # HandOff, one episode, saves rollout.gif
python rollout.py --episodes 5          # average delivery over 5 episodes
python rollout.py --device cpu --out demo.gif
python rollout.py --hold_scale 20       # more decisive grabbing (see below)
```

Expected output:

```
episode 0: reward=[...] delivery_score=0.25
saved rollout.gif
mean delivery_score over 1 episode(s): 0.250
```

`delivery_score` is the fraction of items delivered to the target zone
(0–1). It is stochastic (the policy samples), so it varies across seeds;
average a few episodes for a stable number.

## Use the policy in your own code

```python
from diffusion_policy import DiffusionPolicy

policy = DiffusionPolicy("dp_handoff_robot1.pt", device="cpu")
policy.reset()                       # once per episode

# `state` is this robot's 121-dim observation, produced by the MovingOut
# StatesEncoder:  encoder.get_state_by_current_obs_states(env.get_all_states())
action = policy.get_action(state)    # -> [forward_backward, angle_rad, hold]

# Asynchronous runners can instead receive the complete 8-step horizon and
# align it using their own control-step clock.
plan = policy.get_action_plan(state) # -> 8 env-native actions
```

- **Observation**: 121-dim per robot (from `moving_out.utils.states_encoding.StatesEncoder`).
- **Action**: `[forward_backward ∈ [-1,1], angle in radians, hold (bool)]`,
  fed straight to `env.step([action_robot1, action_robot2])`.
- The policy predicts a short action sequence and executes a few steps per
  inference (re-planning automatically), so call `get_action` once per env
  step and call `reset()` at the start of each episode.
- `get_action_plan` returns the complete predicted horizon without using the
  local two-action cache. Do not call both APIs for the same control step.

## Notes

- **`hold_scale`** (default 5) controls how decisively the policy grabs and
  releases: it is the temperature on the grab/release logits
  (`softmax(logits * hold_scale)`). Higher = more decisive; `<= 0` = argmax
  (deterministic grab). Because the grab head is regressed rather than
  calibrated, this knob noticeably changes delivery — try `--hold_scale 20`
  if the robots grab or drop erratically.
- These models are map-specific (trained on HandOff); they will not transfer
  to other maps.
