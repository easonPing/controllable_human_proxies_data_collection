"""Roll out two diffusion policies in the MovingOut environment.

Each robot is driven by its own diffusion checkpoint. This is an INFERENCE-ONLY
demo: it loads the two shipped models, runs one or more episodes, prints the
reward and the delivery score, and (optionally) saves a GIF.

    python rollout.py                       # HandOff, the two shipped models
    python rollout.py --episodes 3 --out handoff.gif
    python rollout.py --map HandOff --hold_scale 20

Requires the MovingOut environment to be installed (``pip install -e`` the
moving_out_env package); see README.md.
"""
import argparse
import sys

import numpy as np
import torch

from diffusion_policy import DiffusionPolicy


def run_episode(env, encoder, p1, p2, max_steps, seed, render):
    env.reset(map_name=env_map(env), add_noise_to_item=False)
    try:
        env.seed(seed)
    except Exception:
        pass
    p1.reset()
    p2.reset()
    torch.manual_seed(seed)

    frames = []
    total_reward = np.zeros(2)
    for _ in range(max_steps):
        states = encoder.get_state_by_current_obs_states(env.get_all_states())
        a1 = p1.get_action(states[0])
        a2 = p2.get_action(states[1])
        out = env.step([a1, a2])
        rew = out[1]
        done = bool(out[2]) or bool(out[3]) if len(out) >= 4 else bool(out[2])
        total_reward += np.asarray(rew, dtype=float)
        if render:
            frames.append(np.asarray(env.render("rgb_array")))
        if env.global_score() >= 0.999 or done:
            break
    return total_reward, float(env.global_score()), frames


def env_map(env):
    return getattr(env, "_deploy_map", None)


def save_gif(frames, path):
    if not frames:
        return
    step = max(1, len(frames) // 80)
    frames = frames[::step]
    try:
        import imageio
        imageio.mimsave(path, frames, duration=0.08, loop=0)
    except Exception:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=80, loop=0)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="HandOff", help="map name")
    ap.add_argument("--robot1", default="dp_handoff_robot1.pt")
    ap.add_argument("--robot2", default="dp_handoff_robot2.pt")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--hold_scale", type=float, default=5.0,
                    help="grab decisiveness (softmax temperature on the hold "
                         "logits); higher = more decisive, <= 0 = argmax")
    ap.add_argument("--device", default=None, help="cpu | cuda | cuda:0 ...")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rollout.gif",
                    help="GIF path; empty string disables rendering")
    args = ap.parse_args()

    try:
        from moving_out.benchmarks.moving_out import MovingOutEnv
        from moving_out.utils.states_encoding import StatesEncoder
    except ImportError:
        sys.exit("MovingOut env not found — install it first (see README.md).")

    p1 = DiffusionPolicy(args.robot1, device=args.device, hold_scale=args.hold_scale)
    p2 = DiffusionPolicy(args.robot2, device=args.device, hold_scale=args.hold_scale)

    env = MovingOutEnv(use_state=False, map_name=args.map)
    env._deploy_map = args.map
    encoder = StatesEncoder(max_number_shaps=7)

    scores = []
    for ep in range(args.episodes):
        render = bool(args.out) and ep == 0
        reward, score, frames = run_episode(
            env, encoder, p1, p2, args.max_steps, args.seed + ep, render)
        scores.append(score)
        print(f"episode {ep}: reward={reward.round(2).tolist()} "
              f"delivery_score={score:.2f}")
        if render and frames:
            save_gif(frames, args.out)
    print(f"mean delivery_score over {args.episodes} episode(s): "
          f"{np.mean(scores):.3f}")


if __name__ == "__main__":
    main()
