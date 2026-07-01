"""V-JEPA2-AC: a frozen V-JEPA2 video encoder + an action-conditioned predictor.

A new architecture/paradigm for the Phase-4 zoo. It satisfies the SAME world-model
interface every other model implements (`encode`/`predict`/`rollout`/`get_cost`),
so it slots into both the GCS-Align metric and the CEM-MPC eval unmodified.

Design (lowest-risk, interface-exact): subclass :class:`LeWM` and override **only**
``encode``. Everything action-conditioned — ``predict`` (action enters the
predictor via AdaLN-zero), ``rollout`` (advance the latent one action-block at a
time), ``criterion`` / ``get_cost`` (differentiable goal-conditioned latent MSE) —
is inherited verbatim from LeWM. The only new code is the encoder adapter that
turns V-JEPA2's tubelet *video* encoder into LeWM's expected per-frame vector
latent:

* V-JEPA2 (``transformers.VJEPA2Model``) is a video model: it takes
  ``pixel_values_videos`` of shape ``(N, num_frames, C, H, W)``, embeds tubelets of
  ``tubelet_size`` frames, has **no CLS token**, and does not accept
  ``interpolate_pos_encoding``. We encode each frame independently as a minimal
  ``tubelet_size``-frame clip (repeat the frame across time), call the encoder with
  ``skip_predictor=True`` to get the patch tokens ``(N, n_patches, hidden)``, and
  **mean-pool over patches** to a per-frame vector ``(N, hidden)`` — exactly LeWM's
  ``(B, T, D)`` latent, so all inherited LeWM code is unchanged.

The encoder is frozen (PreJEPA-style); only the projector + action-conditioned
predictor train.
"""

import torch
from einops import rearrange

from stable_worldmodel.wm.lewm.lewm import LeWM
from stable_worldmodel.wm.prejepa.prejepa import PreJEPA


class VJEPA2AC(LeWM):
    """Frozen V-JEPA2 encoder (used per-frame) + LeWM action-conditioned predictor.

    Args:
        encoder: a ``transformers.VJEPA2Model`` (e.g. from
            ``stable_worldmodel.wm.prejepa.module.create_backbone('vjepa2_large')``).
        predictor / action_encoder / projector / pred_proj: as in :class:`LeWM`.
        history_size: number of context frames (= ``wm.history_size``).
        tubelet_size: frames per V-JEPA2 tubelet (2 for the released checkpoints);
            each frame is repeated this many times to form a minimal clip.
        pool: ``'mean'`` (over patch tokens; V-JEPA2 has no CLS) or ``'first'``.
        freeze_encoder: freeze the V-JEPA2 backbone (default True).
    """

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        history_size=3,
        tubelet_size=2,
        pool='mean',
        freeze_encoder=True,
        **kwargs,
    ):
        super().__init__(
            encoder, predictor, action_encoder, projector, pred_proj, **kwargs
        )
        self.history_size = history_size
        self.tubelet_size = tubelet_size
        self.pool = pool
        if freeze_encoder:
            self.encoder.eval()
            self.encoder.requires_grad_(False)

    def _encode_frames(self, frames):
        """frames: (N, C, H, W) -> per-frame latent (N, hidden).

        Each frame is encoded as a minimal ``tubelet_size``-frame clip; the frozen
        V-JEPA2 encoder returns patch tokens, which we mean-pool (no CLS token).
        """
        clip = frames.unsqueeze(1).repeat(
            1, self.tubelet_size, 1, 1, 1
        )  # (N, tubelet, C, H, W)
        with torch.no_grad():
            out = self.encoder(pixel_values_videos=clip, skip_predictor=True)
        tok = out.last_hidden_state  # (N, n_patches, hidden)
        return tok.mean(dim=1) if self.pool == 'mean' else tok[:, 0]

    def encode(self, info):
        """Encode observations (and actions) into embeddings.

        Mirrors ``LeWM.encode``'s tensor convention exactly (so the inherited
        ``rollout``/``get_cost`` pipeline is unchanged), swapping the per-frame
        feature extractor for the frozen V-JEPA2 encoder.
        """
        pixels = info['pixels'].to(next(self.encoder.parameters()).dtype)
        b = pixels.size(0)
        frames = rearrange(pixels, 'b t ... -> (b t) ...')  # (N, C, H, W)
        feat = self._encode_frames(frames)  # (N, hidden)
        emb = self.projector(feat)  # (N, D)
        info['emb'] = rearrange(emb, '(b t) d -> b t d', b=b)

        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])

        return info


class VJEPA2ACSpatial(PreJEPA):
    """Frozen V-JEPA2 ViT-L encoder (PER-PATCH tokens) + PreJEPA action-conditioned predictor.

    The Phase-4b :class:`VJEPA2AC` mean-pools V-JEPA2's patch tokens into a single
    per-frame vector (LeWM-style). An audit traced both of V-JEPA2-AC's GCS↔success
    outliers (Reacher & Push-T) to that mean-pool: V-JEPA2-L spreads task-object
    information across ~188/256 patch tokens, so uniform averaging washes out the
    small task object (block / fingertip) and yields an uninformative latent cost.

    This class RE-BASES the architecture onto :class:`PreJEPA`, which natively
    carries a per-patch ``P`` axis through ``encode``/``predict``/``rollout``/
    ``criterion``/``get_cost`` and concatenates the action across patches. We
    override **only** the image encoder so it returns ALL 256 patch tokens
    ``(B, T, P=256, D=1024)`` from the frozen V-JEPA2 ViT-L (instead of a pooled
    vector); everything else is inherited verbatim from PreJEPA.

    V-JEPA2 (``transformers.VJEPA2Model``) is a video model: it takes
    ``pixel_values_videos`` of shape ``(N, num_frames, C, H, W)``, embeds tubelets of
    ``tubelet_size`` frames, has **no CLS token**, and does not accept
    ``interpolate_pos_encoding``. We encode each frame independently as a minimal
    ``tubelet_size``-frame clip (repeat the frame across time) with
    ``skip_predictor=True`` and keep every patch token.

    Args:
        encoder: a ``transformers.VJEPA2Model`` (e.g. from
            ``create_backbone('vjepa2_large')``).
        predictor / extra_encoders / history_size / num_pred /
            interpolate_pos_encoding: as in :class:`PreJEPA`.
        tubelet_size: frames per V-JEPA2 tubelet (2 for the released checkpoints);
            each frame is repeated this many times to form a minimal clip.
        freeze_encoder: freeze the V-JEPA2 backbone (default True).
    """

    def __init__(
        self,
        encoder,
        predictor,
        extra_encoders=None,
        tubelet_size=2,
        freeze_encoder=True,
        **kwargs,
    ):
        super().__init__(
            encoder, predictor, extra_encoders=extra_encoders, **kwargs
        )
        self.tubelet_size = tubelet_size
        if freeze_encoder:
            self.backbone.eval()
            self.backbone.requires_grad_(False)

    def _encode_image(self, pixels):
        """(B, T, C, H, W) -> per-patch tokens (B, T, 256, 1024).

        Each frame is encoded as a minimal ``tubelet_size``-frame clip; the frozen
        V-JEPA2 encoder returns the 256 patch tokens (no CLS token), ALL of which
        are kept (no mean-pool).
        """
        B = pixels.shape[0]
        frames = rearrange(pixels, 'b t c h w -> (b t) c h w')
        clip = frames.unsqueeze(1).repeat(
            1, self.tubelet_size, 1, 1, 1
        )  # (N, tubelet, C, H, W)
        clip = clip.to(next(self.backbone.parameters()).dtype)
        with torch.no_grad():
            out = self.backbone(
                pixel_values_videos=clip, skip_predictor=True
            )
        tok = out.last_hidden_state  # (N, P, D)
        assert tok.shape[1] == 256, tok.shape
        return rearrange(tok.detach().float(), '(b t) p d -> b t p d', b=B)


__all__ = ['VJEPA2AC', 'VJEPA2ACSpatial']
