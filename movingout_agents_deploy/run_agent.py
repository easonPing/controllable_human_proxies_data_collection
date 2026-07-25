"""Run MovingOut agents from this package.

Both robots controlled by agents:
    python run_agent.py --map EfficientRoutes --method PRISM --episodes 3

Options:
    --mode stoch|det   action sampling mode (default: stoch)
    --render           save a GIF of each episode (episode_<n>.gif)

The blue checkpoint drives robot 1, the red checkpoint drives robot 2.
For human-AI play, see `agent_action()` below: query the agent only for
the robot the human is NOT controlling.
"""
import argparse
import os

import torch
from moving_out.benchmarks.moving_out import MovingOutEnv

try:
    from .movingout_agent import MovingOutAgent
except ImportError:  # Preserve direct ``python run_agent.py`` usage.
    from movingout_agent import MovingOutAgent

HERE = os.path.dirname(os.path.abspath(__file__))


def agent_action(agent, env, robot_id, obs, deterministic, generator=None):
    """One robot's env-ready action from its agent."""
    raw = agent.act(obs[robot_id], deterministic=deterministic, generator=generator)
    return agent.env_action(raw, holding=env.if_hold[robot_id])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True,
                    choices=["EfficientRoutes", "SequentialRotations"])
    ap.add_argument("--method", required=True,
                    choices=["GAMMA", "COMEDI", "PRISM"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--mode", choices=["stoch", "det"], default="stoch")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    blue = MovingOutAgent.load(
        os.path.join(HERE, "checkpoints", f"{args.map}-{args.method}-blue.pt"))
    red = MovingOutAgent.load(
        os.path.join(HERE, "checkpoints", f"{args.map}-{args.method}-red.pt"))
    det = args.mode == "det"

    env = MovingOutEnv(map_name=args.map, reward_setting="dense")
    for ep in range(args.episodes):
        g = torch.Generator().manual_seed(ep * 7919 + 13)
        env.reset()
        obs = env.get_state()
        frames, steps = [], 0
        for t in range(500):
            acts = [agent_action(blue, env, 0, obs, det, g),
                    agent_action(red, env, 1, obs, det, g)]
            _, _, done, *_ = env.step(acts)
            obs = env.get_state()
            steps = t + 1
            if args.render:
                frames.append(env.render(mode="rgb_array"))
            if done or env.global_score() >= 0.999:
                break
        print(f"episode {ep}: steps={steps} score={env.global_score():.2f}")
        if args.render and frames:
            from PIL import Image
            imgs = [Image.fromarray(f) for f in frames[:: max(1, len(frames) // 70)]]
            imgs[0].save(f"episode_{ep}.gif", save_all=True,
                         append_images=imgs[1:], duration=75, loop=0)


if __name__ == "__main__":
    main()
