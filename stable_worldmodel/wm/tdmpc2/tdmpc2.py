import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from .module import (
    two_hot,
    two_hot_inv,
    log_std,
    gaussian_logprob,
    squash,
    SimNorm,
    NormedLinear,
    RunningScale,
    mlp,
    weight_init,
    zero_init,
)


class TDMPC2(nn.Module):
    """
    Main Neural Network Architecture for TD-MPC2.
    Handles dynamic encoding of modalities, latent dynamics, reward prediction, and action planning.

    Encoder takes observations only.

    Args:
        cfg: Configuration object containing model and training hyperparameters.
        extra_encoders: Optional pre-built ModuleDict of observation encoders.
            If provided, these are used directly instead of building default MLP
            encoders from cfg. Allows injecting custom encoder architectures
            (e.g. CNNs, transformers) without modifying this class.
            Output dims must match cfg.wm.encoding values.

    Assumptions:
        - Continuous Control: The algorithm assumes continuous action spaces.
        - Action Bounds: Actions are strictly assumed to be normalized to the range [-1.0, 1.0].
            The actor network and MPPI planner enforce this bound via Tanh and clamping.
        - Reward Scaling: Environment rewards and Q-values should fall roughly within the
            [vmin, vmax] range defined in the config, as they are discretized using two-hot encoding.
    """

    def __init__(self, cfg, extra_encoders: nn.ModuleDict | None = None):
        super().__init__()
        self.cfg = cfg
        self.scale = RunningScale(cfg.wm.tau)

        # --- SimNorm toggle (the "regularizer-off" TD-MPC1 ablation) ----------
        # TD-MPC2's SimNorm (a per-simplex softmax on the encoder output AND the
        # dynamics activation) is the primary collapse-PREVENTING regularizer.
        # The hypothesis (see tdmpc1_vision.yaml) is that on a low-anchor task
        # SimNorm SATURATES and kills the planning action-gradients. Setting
        # ``cfg.wm.use_simnorm=False`` turns it into TD-MPC1 (no SimNorm): we
        # replace SimNorm with a plain LayerNorm in BOTH the encoder output and
        # the dynamics activation — keeping the latent magnitude bounded (so the
        # latent-MSE planning cost stays well-scaled) WITHOUT the softmax
        # simplex that saturates. DEFAULTS to True == the unchanged TD-MPC2
        # behavior, so existing checkpoints/jobs are unaffected. (A ``getattr``
        # fallback elsewhere covers pickled models predating this flag.)
        self.use_simnorm = cfg.wm.get('use_simnorm', True)

        encoding_cfg = cfg.wm.get('encoding', {})
        self.use_pixels = 'pixels' in encoding_cfg
        self.latent_dim = 0

        if self.use_pixels:
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, 7, stride=2),
                nn.Mish(),
                nn.Conv2d(32, 32, 5, stride=2),
                nn.Mish(),
                nn.Conv2d(32, 32, 3, stride=2),
                nn.Mish(),
                nn.Conv2d(32, 32, 3, stride=1),
                nn.Mish(),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 3, cfg.image_size, cfg.image_size)
                cnn_out_dim = self.cnn(dummy).shape[1]

            pixel_dim = encoding_cfg['pixels']
            self.pixel_encoder = nn.Linear(cnn_out_dim, pixel_dim)
            self.latent_dim += pixel_dim

        if extra_encoders is not None:
            self.extra_encoders = extra_encoders
        else:
            # Default: build a two-layer MLP encoder for each non-pixel modality
            self.extra_encoders = nn.ModuleDict()
            for key, out_dim in encoding_cfg.items():
                if key == 'pixels':
                    continue
                in_dim = cfg.extra_dims[key]
                self.extra_encoders[key] = nn.Sequential(
                    NormedLinear(in_dim, cfg.wm.enc_dim),
                    nn.Linear(cfg.wm.enc_dim, out_dim),
                    nn.LayerNorm(out_dim),
                )

        # Accumulate latent dim from all non-pixel encoders
        for key, out_dim in encoding_cfg.items():
            if key != 'pixels':
                self.latent_dim += out_dim

        assert self.latent_dim > 0, (
            'Model must have pixels or at least one extra_encoder defined.'
        )

        # Latent normalizer applied to BOTH the encoder output and (as the
        # final dynamics activation). SimNorm when use_simnorm (TD-MPC2), else a
        # plain LayerNorm over the full latent (TD-MPC1 "regularizer-off"). The
        # encoder one is a stored module (``sim_norm`` name kept so the train
        # script's enc-optimizer regex and any checkpoint keys still match —
        # LayerNorm adds learnable params under that name, which is fine).
        self.sim_norm = self._make_latent_norm(cfg)

        # Latent dynamics model: predicts next latent state z' from (z, a).
        # Same normalizer family as the encoder for its final activation.
        self.dynamics = mlp(
            self.latent_dim + cfg.action_dim,
            cfg.wm.mlp_dim,
            self.latent_dim,
            act=self._make_latent_norm(cfg),
        )

        # Reward predictor: predicts expected reward from (z, a) as a two-hot distribution
        self.reward = mlp(
            self.latent_dim + cfg.action_dim, cfg.wm.mlp_dim, cfg.wm.num_bins
        )

        # Policy prior (actor): outputs (mean, log_std) of a Gaussian over actions given z.
        # Used both to compute the policy loss and to warm-start CEM planning.
        self.pi = mlp(self.latent_dim, cfg.wm.mlp_dim, 2 * cfg.action_dim)

        # Ensemble of Q-functions: each predicts action-value from (z, a) as a two-hot
        # distribution. An ensemble is used for clipped double-Q to reduce overestimation.
        self.qs = nn.ModuleList(
            [
                mlp(
                    self.latent_dim + cfg.action_dim,
                    cfg.wm.mlp_dim,
                    cfg.wm.num_bins,
                    dropout=0.01,
                )
                for _ in range(cfg.wm.num_q)
            ]
        )
        self.target_qs = deepcopy(self.qs)
        for p in self.target_qs.parameters():
            p.requires_grad = False

        # Weight initialization (matches official TD-MPC2)
        self.apply(weight_init)
        zero_init([self.reward[-1].weight])
        for q in self.qs:
            zero_init([q[-1].weight])
        for q in self.target_qs:
            zero_init([q[-1].weight])

        # --- Goal-conditioned latent-MSE planning support ---------------------
        # When the model is goal-conditioned by concatenating the goal into a
        # state encoding key (cfg.goal_obs_key, e.g. 'state' for TwoRoom: the
        # train script augments state s -> [s, g]), planning/probing in this repo
        # uses a *differentiable goal-MSE* cost in the AUGMENTED latent space
        # (see ``get_cost_goal_mse``) instead of the native reward/value cost.
        # The dataset state/goal columns are RAW (un-normalized) at eval time
        # (the plan config does not z-score 'state'/'goal_state'), but training
        # z-scores the augmented state. We therefore stash the training z-score
        # stats of the augmented state as buffers so ``get_cost_goal_mse`` can
        # normalize the states it constructs from the raw info_dict identically.
        # ``state_mean/std`` cover the FULL augmented vector ([s, g], dim 2*D);
        # registered as length-0 placeholders so they round-trip through pickle
        # and ``load_state_dict`` even before ``set_state_norm`` is called.
        self.goal_obs_key = cfg.get('goal_obs_key')
        # Opt out of CEM actor warm-start so planning is initialized identically
        # to the (non-Actionable) JEPA models — zero-padded, NO RL policy prior
        # at inference (the shared cross-paradigm protocol). Override via
        # cfg.no_actor_warmstart=False to use the TD-MPC2 actor warm-start.
        self.no_actor_warmstart = cfg.get('no_actor_warmstart', True)
        self.register_buffer(
            'state_mean', torch.zeros(0, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            'state_std', torch.ones(0, dtype=torch.float32), persistent=True
        )

    def _make_latent_norm(self, cfg) -> nn.Module:
        """Build the latent normalizer: SimNorm (TD-MPC2) or LayerNorm (TD-MPC1).

        When ``cfg.wm.use_simnorm`` (default True) this is the standard TD-MPC2
        SimNorm (per-simplex softmax). When False — the TD-MPC1 "regularizer-off"
        ablation — it is a plain ``LayerNorm`` over the full latent dimension,
        which bounds the latent scale (so the latent-MSE planning cost stays
        well-conditioned) WITHOUT the softmax simplex that saturates the
        action-gradients. Used identically for the encoder output and the
        dynamics' final activation.
        """
        if getattr(self, 'use_simnorm', True):
            return SimNorm(cfg)
        return nn.LayerNorm(self.latent_dim)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Resize the (possibly length-0) norm buffers to the checkpoint's.

        ``state_mean``/``state_std`` are registered as length-0 placeholders, so
        a fresh model loaded via ``instantiate(config) + load_state_dict`` (the
        JEPA path) would otherwise hit a size mismatch against the saved length-
        ``2*D`` stats. Resize to match before the standard copy runs.
        """
        for name in ('state_mean', 'state_std'):
            key = prefix + name
            if key in state_dict:
                buf = getattr(self, name)
                src = state_dict[key]
                if buf is not None and buf.shape != src.shape:
                    setattr(
                        self,
                        name,
                        torch.empty_like(src, device=buf.device),
                    )
        return super()._load_from_state_dict(
            state_dict, prefix, *args, **kwargs
        )

    def set_state_norm(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Persist the training z-score stats of the AUGMENTED state.

        Called once by the train script after computing the goal-augmented
        state column's mean/std (over the full ``[obs, goal]`` vector). These
        let ``get_cost_goal_mse`` normalize the raw eval-time ``state``/
        ``goal_state`` exactly as training did. Stored as buffers so they are
        carried by both ``torch.save(model)`` and ``state_dict`` checkpoints.
        """
        mean = torch.as_tensor(mean, dtype=torch.float32).flatten()
        std = torch.as_tensor(std, dtype=torch.float32).flatten()
        self.state_mean = mean
        self.state_std = std

    def encode(self, obs_dict: dict) -> torch.Tensor:
        """Encode observations into a latent state, normalized by ``sim_norm``.

        ``sim_norm`` is SimNorm (TD-MPC2, the default) or a plain LayerNorm
        (TD-MPC1 "regularizer-off", ``cfg.wm.use_simnorm=False``).

        Handles arbitrary leading dimensions — (B,), (B, T), (B, N) — by
        flattening into the batch axis per modality and restoring afterward.
        """
        embeddings = []
        target_dtype = next(self.parameters()).dtype

        # Process primary vision modality — flatten all leading dims into batch
        if self.use_pixels:
            obs = obs_dict['pixels'].to(target_dtype)
            if obs.shape[-1] == 3:
                obs = obs.movedim(-1, -3)
            lead_dims = obs.shape[:-3]  # e.g. (B,) or (B, T)
            obs_flat = obs.reshape(
                -1, *obs.shape[-3:]
            )  # (prod(lead), C, H, W)
            cnn_out = self.cnn(obs_flat)
            z_pixels = self.pixel_encoder(cnn_out).view(*lead_dims, -1)
            embeddings.append(z_pixels)

        # Process extra modalities (state, proprioception, etc.)
        for key, encoder in self.extra_encoders.items():
            obs = obs_dict[key].to(target_dtype)  # (*lead, dim)
            lead = obs.shape[:-1]
            obs_flat = obs.reshape(-1, obs.shape[-1])  # (prod(lead), dim)
            z = encoder(obs_flat).view(*lead, -1)  # (*lead, enc_dim)
            embeddings.append(z)

        z_concat = torch.cat(embeddings, dim=-1)
        return self.sim_norm(z_concat)

    def forward(
        self, z: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One-step world model prediction.

        Given a latent state and action, predicts the next latent state via the
        dynamics model and the expected reward as a two-hot logit vector.

        Args:
            z: Current latent state of shape (B, latent_dim).
            action: Action of shape (B, action_dim).

        Returns:
            Tuple of (next_z, reward_logits) with shapes (B, latent_dim) and
            (B, num_bins) respectively.
        """
        z_a = torch.cat([z, action], dim=-1)
        return self.dynamics(z_a), self.reward(z_a)

    def rollout(
        self, z: torch.Tensor, horizon: int, num_trajs: int = 1
    ) -> torch.Tensor:
        """Roll out the actor policy from a latent state for a given horizon.

        Samples ``num_trajs`` stochastic trajectories and returns their mean.

        Args:
            z: Initial latent state of shape (B, latent_dim).
            horizon: Number of steps to unroll.
            num_trajs: Number of independent trajectories to average.

        Returns:
            Mean action sequence of shape (B, horizon, action_dim).
        """
        trajs = []
        for _ in range(num_trajs):
            curr_z, traj = z, []
            for _ in range(horizon):
                mean_raw, log_std_raw = self.pi(curr_z).chunk(2, dim=-1)
                act = torch.tanh(
                    mean_raw
                    + log_std(log_std_raw, low=-10, dif=12).exp()
                    * torch.randn_like(mean_raw)
                )
                traj.append(act)
                curr_z = self.dynamics(torch.cat([curr_z, act], dim=-1))
            trajs.append(torch.stack(traj, dim=1))  # (B, horizon, action_dim)
        return torch.stack(trajs).mean(0)  # (B, horizon, action_dim)

    def get_action(
        self,
        info_dict: dict,
        horizon: int = 1,
        prefix_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample an action sequence from the actor policy via latent rollout.

        Encodes the current observation into a latent state, optionally advances
        it through ``prefix_actions`` via the dynamics model, then calls
        ``rollout`` for ``horizon`` steps.

        Args:
            info_dict: Dictionary containing environment state information with
                shape (B, ...).
            horizon: Number of steps to plan.
            prefix_actions: Optional warm-start actions of shape
                (B, t, action_dim) with t < horizon. The latent state is
                advanced through these steps before the actor rollout.

        Returns:
            Action tensor of shape (B, horizon, action_dim).
        """
        device = next(self.parameters()).device
        key = self.goal_obs_key or 'state'
        has_goal = (f'goal_{key}' in info_dict) or ('goal' in info_dict)

        if has_goal:
            # Goal-conditioned warm-start: the actor encodes the SAME augmented,
            # z-scored current state ([state, goal]) the goal-MSE cost uses, so
            # the CEM warm-start is consistent with the planning objective.
            aug_current, _ = self._build_goal_conditioned_states(
                info_dict, device
            )
            z = self.encode({key: aug_current})
        else:
            encoding_keys = list(self.cfg.wm.get('encoding', {}).keys())
            obs_dict = {k: info_dict[k].to(device) for k in encoding_keys}
            z = self.encode(obs_dict)

        if prefix_actions is not None:
            for t in range(prefix_actions.shape[1]):
                z = self.dynamics(
                    torch.cat([z, prefix_actions[:, t].to(device)], dim=-1)
                )

        num_trajs = self.cfg.get('num_pi_trajs', 1)
        return self.rollout(z, horizon, num_trajs)  # (B, horizon, action_dim)

    # ------------------------------------------------------------------ #
    #  Goal-conditioned latent-MSE planning cost (used for PLAN + PROBE)  #
    # ------------------------------------------------------------------ #

    def _norm_aug_state(self, x: torch.Tensor) -> torch.Tensor:
        """Z-score an augmented-state tensor with the stored training stats.

        No-op (identity) if ``set_state_norm`` was never called (empty buffers).
        Broadcasts over arbitrary leading dims; normalizes the last dim.
        """
        if self.state_mean.numel() == 0:
            return x
        mean = self.state_mean.to(device=x.device, dtype=x.dtype)
        std = self.state_std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def _build_goal_conditioned_states(
        self, info_dict: dict, device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct (augmented current, augmented goal-reached) raw states.

        The model is goal-conditioned by concatenating the goal into the
        ``goal_obs_key`` encoding modality (TwoRoom: ``state``). At eval/probe
        time the info_dict carries the RAW current state under ``state`` and the
        RAW goal under ``goal_state`` (env ``_get_info``) — both with whatever
        leading dims the caller passes, e.g. ``(B, N, D)`` from the CEM solver.

          current      = [state,      goal_state]   (the obs the planner sees)
          goal-reached = [goal_state, goal_state]   (constructed goal target)

        Both are z-scored with the training stats. Returns two tensors with the
        same leading dims as the inputs and last dim ``2 * D``.
        """
        key = self.goal_obs_key or 'state'
        assert key in info_dict, (
            f"goal-MSE cost needs '{key}' in info_dict; got {list(info_dict)}"
        )
        # Resolve the goal key. Priority:
        #   1) cfg.goal_col (Cube): the GOAL state lives in its own per-step
        #      column (e.g. 'target' = the 28-d goal obs). The eval framework
        #      broadcasts that column into the info_dict under its own name, so
        #      the planner sees the start-state goal directly (no goal_<key>).
        #   2) 'goal_<key>' (TwoRoom): the episode-final obs of the encoding
        #      column, exposed by World._extract_init_goal as e.g. 'goal_state'.
        #   3) bare 'goal' (JEPA-style image goal fallback).
        goal_col_cfg = self.cfg.get('goal_col')
        if goal_col_cfg is not None and goal_col_cfg in info_dict:
            goal_key = goal_col_cfg
        else:
            goal_key = f'goal_{key}'
            if goal_key not in info_dict:
                goal_key = 'goal' if 'goal' in info_dict else None
        assert goal_key is not None, (
            f"goal-MSE cost needs '{goal_col_cfg or f'goal_{key}'}' (or 'goal') "
            f'in info_dict; got {list(info_dict)}'
        )

        cur = info_dict[key].to(device).float()
        goal = info_dict[goal_key].to(device).float()

        aug_current = torch.cat([cur, goal], dim=-1)
        aug_goal = torch.cat([goal, goal], dim=-1)
        return self._norm_aug_state(aug_current), self._norm_aug_state(
            aug_goal
        )

    def get_cost_goal_mse(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Differentiable goal-conditioned latent-MSE planning cost.

        Mirrors the JEPA template (pldm.get_cost): encode the goal-reached
        augmented state, roll the latent dynamics forward under each candidate
        action trajectory, and return the squared latent distance of the final
        rolled latent to the (detached) goal latent.

          z_init = encode([state, goal])          # augmented current
          z_goal = encode([goal,  goal]).detach()  # augmented goal-reached
          z_t+1  = dynamics(cat[z_t, a_t])
          cost   = sum_d (z_H - z_goal)^2          -> (B, N), lower = better

        Differentiable wrt ``action_candidates``. The reward/value/actor heads
        are NOT used here — this is the pure planning objective the v3 probe
        differentiates. Shape: action_candidates (B, N, H, A) -> cost (B, N).
        """
        device = action_candidates.device
        B, N, H, A = action_candidates.shape

        aug_current, aug_goal = self._build_goal_conditioned_states(
            info_dict, device
        )
        # encode handles arbitrary leading dims; pass through the state encoder.
        z = self.encode({(self.goal_obs_key or 'state'): aug_current})
        z_goal = self.encode({(self.goal_obs_key or 'state'): aug_goal})

        # Collapse leading dims to (B*N, latent_dim).
        z = self._flatten_latent(z, B, N)
        z_goal = self._flatten_latent(z_goal, B, N).detach()

        # Unpack the frameskip (action_block) packing: the shared CEM planner
        # uses a per-horizon-step action of dim A = base_action_dim * block (10
        # for TwoRoom: base 2 x block 5), matching the JEPA models. TD-MPC2's
        # dynamics is single-step (trained at base action_dim), so we roll it
        # base over H*block sub-steps -> the SAME env-step horizon as JEPA.
        sub_steps = self._unpack_actions(action_candidates).reshape(
            B * N, -1, self.cfg.action_dim
        )  # (B*N, H*block, base_action_dim)
        for t in range(sub_steps.shape[1]):
            z = self.dynamics(torch.cat([z, sub_steps[:, t]], dim=-1))

        cost = (z - z_goal).pow(2).sum(dim=-1)  # (B*N,)
        return cost.view(B, N)

    # ------------------------------------------------------------------ #
    #  VISION-ONLY goal-IMAGE-MSE planning cost (cross-paradigm matched)  #
    # ------------------------------------------------------------------ #

    def _encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode a pixel tensor through the CNN path, resizing to image_size.

        Mirrors ``encode``'s pixel branch but is robust to the spatial size of
        the incoming images: the eval/probe pipeline ImageNet-normalizes and
        resizes pixels to ``cfg.eval.img_size`` (which may be the JEPA-default
        224, not TD-MPC2's native 64). The CNN/pixel_encoder were built for
        ``cfg.image_size`` (64), so we bilinearly resize any mismatched input to
        that size before the CNN — keeping the model self-contained regardless
        of the eval transform's image size. No-op when already at image_size.

        Handles arbitrary leading dims; returns ``(*lead, pixel_dim)``.
        """
        target_dtype = next(self.parameters()).dtype
        obs = pixels.to(target_dtype)
        if obs.shape[-1] == 3:  # channels-last -> channels-first
            obs = obs.movedim(-1, -3)
        lead_dims = obs.shape[:-3]
        obs_flat = obs.reshape(-1, *obs.shape[-3:])  # (prod(lead), C, H, W)
        img_size = int(self.cfg.get('image_size', obs_flat.shape[-1]))
        if obs_flat.shape[-1] != img_size or obs_flat.shape[-2] != img_size:
            obs_flat = F.interpolate(
                obs_flat,
                size=(img_size, img_size),
                mode='bilinear',
                align_corners=False,
            )
        cnn_out = self.cnn(obs_flat)
        z_pixels = self.pixel_encoder(cnn_out)
        z_pixels = self.sim_norm(z_pixels)
        return z_pixels.view(*lead_dims, -1)

    def get_cost_goal_image_mse(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Differentiable PLAIN pixel goal-image-MSE planning cost (vision-only).

        The cross-paradigm-matched objective: identical in spirit to the JEPA
        template (``pldm.get_cost``) — encode the goal IMAGE into the goal
        latent, roll the latent dynamics forward under each candidate action
        trajectory, and return the squared latent distance of the final rolled
        latent to the (detached) goal latent. NO augmented state, NO privileged
        info — purely pixels in, latent-MSE out.

          z_init = encode_pixels(info['pixels'])        # current image latent
          z_goal = encode_pixels(info['goal']).detach()  # goal  image latent
          z_t+1  = dynamics(cat[z_t, a_t])
          cost   = sum_d (z_H - z_goal)^2               -> (B, N), lower=better

        Differentiable wrt ``action_candidates``. The reward/value/actor heads
        are NOT used — this is the pure planning objective the v3 probe
        differentiates. Shape: action_candidates (B, N, H, A) -> cost (B, N).
        """
        assert self.use_pixels, (
            'goal_image_mse cost requires a pixel-encoded model.'
        )
        assert 'pixels' in info_dict and 'goal' in info_dict, (
            "goal_image_mse cost needs 'pixels' and 'goal' images in info_dict; "
            f'got {list(info_dict)}'
        )
        device = action_candidates.device
        B, N, H, A = action_candidates.shape

        z = self._encode_pixels(info_dict['pixels'].to(device))
        z_goal = self._encode_pixels(info_dict['goal'].to(device))

        z = self._flatten_latent(z, B, N)
        z_goal = self._flatten_latent(z_goal, B, N).detach()

        # Unpack the frameskip (action_block) packing so the env-step horizon
        # matches the JEPA models (see get_cost_goal_mse for the rationale).
        sub_steps = self._unpack_actions(action_candidates).reshape(
            B * N, -1, self.cfg.action_dim
        )  # (B*N, H*block, base_action_dim)
        for t in range(sub_steps.shape[1]):
            z = self.dynamics(torch.cat([z, sub_steps[:, t]], dim=-1))

        cost = (z - z_goal).pow(2).sum(dim=-1)  # (B*N,)
        return cost.view(B, N)

    def _unpack_actions(self, action_candidates: torch.Tensor) -> torch.Tensor:
        """Unpack a frameskip-packed action trajectory into single sub-steps.

        ``action_candidates`` is ``(..., H, A)`` with ``A = base * block``. Returns
        ``(..., H * block, base)`` — each horizon step expanded into its ``block``
        consecutive base-dim sub-actions, in time order. When ``A == base``
        (block 1) this is a no-op reshape.
        """
        base = self.cfg.action_dim
        *lead, H, A = action_candidates.shape
        assert A % base == 0, (
            f'packed action dim {A} is not a multiple of base action_dim {base}'
        )
        block = A // base
        return action_candidates.reshape(*lead, H * block, base)

    @staticmethod
    def _flatten_latent(z: torch.Tensor, B: int, N: int) -> torch.Tensor:
        """Reshape an encoded latent to ``(B*N, latent_dim)``.

        The state passed to ``encode`` may carry assorted leading dims from the
        CEM solver / probe: ``(B, N, D)``, ``(B, D)`` (no sample axis yet), or
        ``(B, N, history, D)`` (the probe keeps a length-1 history axis). The
        last dim is always the latent; collapse everything before it.

        * leading product == B*N  -> reshape to (B*N, latent).
        * leading product == B    -> broadcast over N, then (B*N, latent).
        """
        lat = z.shape[-1]
        lead = z.numel() // lat
        if lead == B * N:
            return z.reshape(B * N, lat)
        if lead == B:
            return (
                z.reshape(B, lat)
                .unsqueeze(1)
                .expand(B, N, lat)
                .reshape(B * N, lat)
            )
        raise ValueError(
            f'Unexpected latent state shape: {tuple(z.shape)} (B={B}, N={N})'
        )

    def criterion(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Costable-protocol alias for the active goal-MSE planning cost.

        Routes to the pixel goal-image-MSE cost for the vision-only model and
        to the augmented-state goal-MSE cost otherwise, matching ``get_cost``'s
        dispatch so the v3 probe's cost-sensitivity capture wraps the right one.
        """
        plan_cost = self.cfg.get('plan_cost', None)
        use_image = (plan_cost == 'goal_image_mse') or (
            plan_cost is None and self.use_pixels and 'goal' in info_dict
        )
        if use_image:
            return self.get_cost_goal_image_mse(info_dict, action_candidates)
        return self.get_cost_goal_mse(info_dict, action_candidates)

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the cost of candidate action trajectories.

        Dispatches between three cost paths (all kept working):

          * ``goal_image_mse`` (the VISION-only, cross-paradigm-matched path):
            the differentiable PLAIN pixel goal-image-MSE cost
            (:meth:`get_cost_goal_image_mse`). Selected when ``cfg.plan_cost ==
            'goal_image_mse'`` OR (``cfg.plan_cost`` unset AND the model is
            pixel-encoded AND a ``'goal'`` image is present). This is what the
            vision TD-MPC2 plans/probes with — matched to the JEPA models.
          * ``goal_mse`` (the AUGMENTED-state path, state models only): the
            differentiable goal-conditioned latent-MSE cost in the augmented
            ``[state, goal]`` latent (:meth:`get_cost_goal_mse`). Selected when
            ``cfg.plan_cost == 'goal_mse'`` OR (``cfg.plan_cost`` unset, NOT
            pixel-encoded, AND a goal key is present).
          * ``reward_value`` (the native TD-MPC2 cost): rolls out the world
            model, accumulates discounted predicted rewards, and adds a terminal
            conservative-Q value. Selected when ``cfg.plan_cost ==
            'reward_value'`` or no goal is available.

        Args:
            info_dict: Dictionary containing environment state with shape (B, N, ...).
            action_candidates: Candidate action sequences of shape (B, N, H, A).

        Returns:
            Cost tensor of shape (B, N). Lower is better.
        """
        plan_cost = self.cfg.get('plan_cost', None)
        key = self.goal_obs_key or 'state'
        has_image_goal = 'goal' in info_dict
        has_state_goal = (f'goal_{key}' in info_dict) or ('goal' in info_dict)

        # Vision-only goal-image-MSE: explicit, or auto for a pixel-encoded model
        # presented with a goal image.
        use_image_mse = (plan_cost == 'goal_image_mse') or (
            plan_cost is None and self.use_pixels and has_image_goal
        )
        if use_image_mse:
            return self.get_cost_goal_image_mse(info_dict, action_candidates)

        use_goal_mse = (plan_cost == 'goal_mse') or (
            plan_cost is None and not self.use_pixels and has_state_goal
        )
        if use_goal_mse:
            return self.get_cost_goal_mse(info_dict, action_candidates)

        device = action_candidates.device
        encoding_keys = list(self.cfg.wm.get('encoding', {}).keys())

        obs_dict = {key: info_dict[key].to(device) for key in encoding_keys}
        z = self.encode(obs_dict)

        B, N, H, A = action_candidates.shape

        if z.ndim == 2 and z.shape[0] == B:
            z = z.unsqueeze(1).repeat(1, N, 1).view(B * N, -1)
        elif z.ndim == 3 and z.shape[0] == B and z.shape[1] == N:
            z = z.view(B * N, -1)
        elif z.ndim == 2 and z.shape[0] == B * N:
            pass
        else:
            raise ValueError(f'Unexpected latent state shape: {z.shape}')

        # Unpack frameskip packing to single-step (base) actions, as the
        # dynamics/reward heads expect; no-op when A == base action_dim.
        actions = self._unpack_actions(action_candidates).reshape(
            B * N, -1, self.cfg.action_dim
        )

        G, discount = 0, 1.0
        c = self.cfg.wm.get('uncertainty_penalty', 0.5)
        termination = torch.zeros(
            B * N, 1, dtype=torch.float32, device=z.device
        )

        for t in range(actions.shape[1]):
            z_a = torch.cat([z, actions[:, t]], dim=-1)
            reward = two_hot_inv(self.reward(z_a), self.cfg)
            z = self.dynamics(z_a)
            G = G + discount * (1 - termination) * reward
            discount = discount * self.cfg.wm.get('discount', 0.99)

        mu = self.pi(z).chunk(2, dim=-1)[0]
        action = torch.tanh(mu)
        z_a_term = torch.cat([z, action], dim=-1)

        q_logits = torch.stack([q(z_a_term) for q in self.qs])
        q_values = torch.stack(
            [two_hot_inv(logits, self.cfg) for logits in q_logits]
        )

        q_mean = q_values.mean(dim=0)
        q_std = q_values.std(dim=0)

        penalty = c * q_mean.abs() * q_std
        conservative_q = q_mean - penalty
        total_return = G + discount * (1 - termination) * conservative_q

        return -total_return.view(B, N)


def tdmpc2_forward(self, batch, stage, cfg):
    """Forward pass and loss computation for TD-MPC2.

    Designed to be used as a Lightning ``training_step`` or called directly
    from an online training loop via a context object that implements
    ``self.model`` and ``self.log_dict``.

    Args:
        batch: Dict with keys matching cfg.wm.encoding plus 'action' and 'reward'.
        stage: 'train' or 'validate'. Controls target-network soft update.
        cfg: OmegaConf config with wm.* hyperparameters.

    Returns:
        The batch dict with 'loss' set to the total scalar loss.
    """
    encoding_keys = list(cfg.wm.get('encoding', {}).keys())
    B, T_plus_1 = batch['action'].shape[:2]

    flat_obs_dict = {}
    for key in encoding_keys:
        obs = batch[key]
        flat_obs_dict[key] = obs.reshape(-1, *obs.shape[2:])

    all_z = self.model.encode(flat_obs_dict).reshape(B, T_plus_1, -1)

    z = all_z[:, 0]
    target_zs = all_z[:, 1:]

    loss_consistency, loss_reward, loss_value, loss_pi = 0, 0, 0, 0
    discount = cfg.wm.get('discount', 0.99)
    entropy_coef = cfg.wm.get('entropy_coef', 1e-4)

    for t in range(cfg.wm.horizon):
        action = batch['action'][:, t]
        # Dataset reward is (B, 1); collapse to (B,) so two_hot / the
        # target_q = reward.unsqueeze(1) + ... arithmetic stay 2-D (a trailing
        # singleton would make target_q 3-D and break two_hot's scatter_).
        reward = batch['reward'][:, t].reshape(B)

        next_z_pred, reward_pred = self.model.forward(z, action)

        loss_consistency += F.mse_loss(
            next_z_pred, target_zs[:, t].detach()
        ) * (cfg.wm.rho**t)
        target_reward = two_hot(reward, cfg)
        loss_reward += -(
            target_reward * F.log_softmax(reward_pred, dim=-1)
        ).sum(-1).mean() * (cfg.wm.rho**t)

        with torch.no_grad():
            next_z_for_q = target_zs[:, t].detach()
            mean_raw, log_std_raw = self.model.pi(next_z_for_q).chunk(
                2, dim=-1
            )
            log_std_bounded = log_std(log_std_raw, low=-10, dif=12)
            eps = torch.randn_like(mean_raw)
            next_action_pred = torch.tanh(
                mean_raw + eps * log_std_bounded.exp()
            )

            next_z_a = torch.cat([next_z_for_q, next_action_pred], dim=-1)
            q_indices = random.sample(range(cfg.wm.num_q), 2)
            next_qs = [
                two_hot_inv(self.model.target_qs[i](next_z_a), cfg)
                for i in q_indices
            ]
            next_q_min = torch.min(next_qs[0], next_qs[1])
            target_q = reward.unsqueeze(1) + discount * next_q_min
            target_q_two_hot = two_hot(target_q, cfg)

        z_a = torch.cat([z, action], dim=-1)
        for q in self.model.qs:
            loss_value += -(
                target_q_two_hot * F.log_softmax(q(z_a), dim=-1)
            ).sum(-1).mean() * (cfg.wm.rho**t)

        z_detached = z.detach()
        mean_raw, log_std_raw = self.model.pi(z_detached).chunk(2, dim=-1)
        log_std_bounded = log_std(log_std_raw, low=-10, dif=12)
        eps = torch.randn_like(mean_raw)
        log_prob = gaussian_logprob(eps, log_std_bounded)

        action_pi_raw = mean_raw + eps * log_std_bounded.exp()
        _, action_pi, log_prob = squash(mean_raw, action_pi_raw, log_prob)

        scaled_entropy = -log_prob * cfg.action_dim

        z_pi = torch.cat([z_detached, action_pi], dim=-1)
        try:
            self.model.qs.requires_grad_(False)
            qs_pi = torch.stack(
                [two_hot_inv(q(z_pi), cfg) for q in self.model.qs], dim=0
            )
        finally:
            self.model.qs.requires_grad_(True)

        q_indices = random.sample(range(cfg.wm.num_q), 2)
        q_pi_avg = (qs_pi[q_indices[0]] + qs_pi[q_indices[1]]) / 2.0

        if t == 0:
            self.model.scale.update(q_pi_avg)
        q_pi_normalized = self.model.scale(q_pi_avg)

        step_pi_loss = -(entropy_coef * scaled_entropy + q_pi_normalized)
        loss_pi += step_pi_loss.mean() * (cfg.wm.rho**t)

        z = next_z_pred

    loss_consistency /= cfg.wm.horizon
    loss_reward /= cfg.wm.horizon
    loss_value /= cfg.wm.horizon * cfg.wm.num_q
    loss_pi /= cfg.wm.horizon

    total_loss = (
        cfg.wm.consistency_coef * loss_consistency
        + cfg.wm.reward_coef * loss_reward
        + cfg.wm.value_coef * loss_value
        + loss_pi
    )

    self.log_dict(
        {
            f'{stage}/loss': total_loss,
            f'{stage}/consist': loss_consistency,
            f'{stage}/reward': loss_reward,
            f'{stage}/value': loss_value,
            f'{stage}/policy': loss_pi,
        },
        on_step=True,
        sync_dist=False,
        prog_bar=True,
    )

    if stage == 'train':
        for q, t_q in zip(self.model.qs, self.model.target_qs):
            for p, p_t in zip(q.parameters(), t_q.parameters()):
                p_t.data.lerp_(p.data, cfg.wm.tau)

    batch['loss'] = total_loss
    return batch


__all__ = [
    'TDMPC2',
    'tdmpc2_forward',
    'two_hot',
    'two_hot_inv',
    'log_std',
    'gaussian_logprob',
    'squash',
]
