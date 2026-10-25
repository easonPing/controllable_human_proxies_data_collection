"""Diffusion behavior-cloning policy: observation in, action out.

Load a checkpoint and drive one robot:

    policy = DiffusionPolicy("dp_handoff_robot1.pt")
    policy.reset()
    action = policy.get_action(state)      # state: 121-dim obs of THIS robot
    # action == [forward_backward, angle_rad, hold]  (env-native)

The policy predicts a short action sequence by DDIM denoising and executes a
few of them per inference (re-planning automatically), so just call
``get_action`` once per environment step. Depends on ``torch``, ``diffusers``,
``numpy`` and ``diffusion_model`` — NOT on the moving_out env or the training
code, so it can be embedded anywhere.
"""
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

try:
    from .diffusion_model import ConditionalUnet1D, DEFAULT_MODEL_CONFIG
except ImportError:  # Preserve ``python diffusion_policy.py``/rollout usage.
    from diffusion_model import ConditionalUnet1D, DEFAULT_MODEL_CONFIG


class DiffusionPolicy:
    def __init__(self, checkpoint_path, device=None, hold_scale=5.0,
                 num_inference_steps=None):
        """checkpoint_path: a .pt with {config, state_dict, inference}.
        hold_scale: temperature on the grab/release logits before sampling
            (softmax(logits * hold_scale)); higher = more decisive, <= 0 =
            argmax (deterministic grab). Default 5 reproduces the reported
            numbers; 20 often grabs more reliably.
        num_inference_steps: DDIM steps (<= the training step count); None
            uses the training value.
        """
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)

        cfg = dict(DEFAULT_MODEL_CONFIG)
        cfg.update(ckpt.get("config", {}))
        inf = ckpt.get("inference", {})

        self.model = ConditionalUnet1D(
            input_dim=cfg["input_dim"],
            global_cond_dim=cfg["global_cond_dim"],
            use_obs_encoder=cfg.get("use_obs_encoder", True),
            diffusion_step_embed_dim=cfg["diffusion_step_embed_dim"],
            down_dims=cfg["down_dims"],
            kernel_size=cfg["kernel_size"],
            n_groups=cfg["n_groups"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.previous_steps = inf.get("previous_steps", 1)
        self.pred_horizon = inf.get("pred_horizon", 8)
        self.action_dim = inf.get("action_dim", 5)
        self.selected_actions = inf.get("selected_actions", 2)
        num_train = inf.get("num_train_timesteps", 32)
        if num_inference_steps is None:
            num_inference_steps = num_train
        self.hold_scale = None if (hold_scale is not None and hold_scale <= 0) \
            else hold_scale

        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )
        self.scheduler.set_timesteps(min(num_inference_steps, num_train))
        self.reset()

    def reset(self):
        """Clear the observation history and the pending action plan."""
        self._history = []      # most-recent `previous_steps` states
        self._plan = []         # decoded actions waiting to be executed

    def _push(self, state):
        # mirror the training-time buffer: keep the last `previous_steps`
        # states, pad-filling at episode start
        while len(self._history) < self.previous_steps:
            self._history.append(state)
        if self.previous_steps > 0:
            self._history = self._history[1:] + [state]

    @torch.no_grad()
    def _plan_actions(self, state):
        past = np.asarray(self._history, dtype=np.float32).reshape(-1)
        obs = np.concatenate([past, np.asarray(state, dtype=np.float32)])
        obs = torch.tensor(obs, dtype=torch.float32,
                           device=self.device).unsqueeze(0)

        # denoise an action sequence from noise
        action = torch.empty((1, self.pred_horizon, self.action_dim),
                             device=self.device).normal_(-1, 1)
        for t in self.scheduler.timesteps:
            noise_pred = self.model(action, t, obs)
            action = self.scheduler.step(model_output=noise_pred, timestep=t,
                                         sample=action)[0]

        # action layout per step: [fb, cos, sin, hold_onehot x2]
        fb = torch.clip(action[0, :, 0], -1, 1)
        angle = torch.atan2(action[0, :, 2], action[0, :, 1])
        if self.hold_scale is None:
            holds = torch.argmax(action[0, :, 3:], dim=-1)
        else:
            probs = F.softmax(action[0, :, 3:] * self.hold_scale, dim=-1)
            holds = torch.multinomial(probs, num_samples=1).squeeze(-1)

        plan = []
        for i in range(self.pred_horizon):
            plan.append([float(fb[i]), float(angle[i]), bool(int(holds[i]))])
        return plan

    def get_action_plan(self, state):
        """Return the complete predicted horizon as env-native actions.

        Unlike :meth:`get_action`, this method does not keep a local
        ``selected_actions`` cache. It is intended for asynchronous runners
        that align the full plan against their own control-step clock.
        """
        self._push(state)
        self._plan = []
        return self._plan_actions(state)

    def get_action(self, state):
        """state: this robot's encoded observation (list/np array, 121-dim).
        Returns one env-native action [forward_backward, angle_rad, hold]."""
        if not self._plan:
            self._push(state)
            plan = self._plan_actions(state)
            n = min(self.selected_actions, self.pred_horizon)
            self._plan = plan[:n]
        return self._plan.pop(0)
