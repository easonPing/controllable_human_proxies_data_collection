"""Standalone MovingOut agent — pure PyTorch, no BenchMARL/torchrl needed.

One shared IPPO policy controls BOTH robots: feed each robot its own 121-dim
observation (from moving_out's StatesEncoder / env.get_state()) and apply the
returned action to that robot.

Action semantics (IMPORTANT): these agents were trained with the
"desired-hold" interface: raw_action[2] > 0 means "I want to be holding".
`env_action()` converts that to the environment's toggle button given the
robot's current holding state, and scales the turn command by pi the same
way the PettingZoo wrapper does.

Usage:
    agent = MovingOutAgent.load("er_efficientroutes.pt")
    obs = env.reset()                       # [obs_robot1, obs_robot2]
    raw = agent.act(obs[i], deterministic=True)
    a_i = agent.env_action(raw, holding=env.if_hold[i])
    env.step([a_1, a_2])
"""
import math

import torch
import torch.nn as nn

_LOG_E_MINUS_1 = math.log(math.e - 1.0)  # biased_softplus_1.0 shift


class MovingOutAgent(nn.Module):
    OBS_DIM, HID, ACT_DIM = 121, 256, 3

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(self.OBS_DIM, self.HID), nn.Tanh(),
            nn.Linear(self.HID, self.HID), nn.Tanh(),
            nn.Linear(self.HID, 2 * self.ACT_DIM),
        )

    def forward(self, obs):
        loc, scale_raw = self.mlp(obs).chunk(2, dim=-1)
        scale = nn.functional.softplus(scale_raw + _LOG_E_MINUS_1)
        return loc, scale

    @torch.no_grad()
    def act(self, obs, deterministic=True, generator=None):
        """obs: array-like (121,) for ONE robot -> raw action in [-1,1]^3."""
        x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
        loc, scale = self(x)
        z = loc if deterministic else torch.normal(loc, scale, generator=generator)
        return torch.tanh(z)[0].numpy()

    @staticmethod
    def env_action(raw, holding):
        """Raw policy action -> action for MovingOutEnv.step:
        [forward, turn*pi, hold_button] with desired-hold -> toggle logic."""
        press = bool(raw[2] > 0) != bool(holding)
        return [float(raw[0]), float(raw[1]) * math.pi, press]

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls()
        agent.load_state_dict(payload["state_dict"])
        agent.eval()
        agent.meta = payload.get("meta", {})
        return agent
