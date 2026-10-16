"""PassiveShape: NoisyShape plus a passive, per-episode-static second object.

Object A is the usual action-driven NoisyShape object. Object B (a hollow
ring) gets a random colour and position at reset and then never changes and
never responds to actions, so it provides no inverse-dynamics signal. B's
attributes are exposed as ``info['b_color_id']``, ``info['b_position_x']``
and ``info['b_position_y']`` for probing.

* :class:`PassiveShapeEnv`    - i.i.d. random background frame per step.
* :class:`PassiveShapeSeqEnv` - temporally coherent (sequential) background.

``reward_mode='noise'`` (default, used in the paper) keeps the environment
reward-free. ``reward_mode='b_color'`` sets r_t = b_color_id / 3, a
per-episode constant determined by B (not used for any reported result).

Paper: Appendix F.4, "Passive and non-action-driven objects" (Table 16).
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from .noisy_shape import NoisyShapeEnv, COLORS
from .noisy_shape_seq import NoisyShapeSeqEnv


class _PassiveBMixin:
    """Adds a per-episode-static second object B. Base class supplies A + background."""

    def __init__(self, *args, reward_mode="noise", b_radius_frac=0.11, **kwargs):
        self._reward_mode = reward_mode
        self._b_radius_frac = b_radius_frac
        self._b_color_id = 0
        self._b_pos = (0.5, 0.5)
        super().__init__(*args, **kwargs)

    # ---- B is sampled once per episode, before the base renders --------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        # sample B first so the (base) render + obs include it
        self._b_color_id = int(self._rng.randint(0, len(COLORS)))
        self._b_pos = (float(self._rng.uniform(0.15, 0.85)),
                       float(self._rng.uniform(0.15, 0.85)))
        return super().reset(seed=None, options=options)  # seed=None: don't re-seed rng

    def _render_image(self):
        arr = super()._render_image()          # A on its background (np uint8)
        pil = Image.fromarray(arr)
        draw = ImageDraw.Draw(pil)
        w, h = self._size
        cx, cy = self._b_pos[0] * w, self._b_pos[1] * h
        r = self._b_radius_frac * min(self._size)
        color = COLORS[self._b_color_id]
        # hollow ring -> visually distinct from A's solid shapes ("target marker")
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=3)
        return np.array(pil, dtype=np.uint8)

    def _make_info(self):
        info = super()._make_info()
        info["b_color_id"] = self._b_color_id
        info["b_position_x"] = self._b_pos[0]
        info["b_position_y"] = self._b_pos[1]
        return info

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        if self._reward_mode == "b_color":
            reward = float(self._b_color_id) / (len(COLORS) - 1)   # per-episode constant set by B
        return obs, reward, done, truncated, info


class PassiveShapeEnv(_PassiveBMixin, NoisyShapeEnv):
    """A (action-driven) + passive static B, on a RANDOM per-step background."""


class PassiveShapeSeqEnv(_PassiveBMixin, NoisyShapeSeqEnv):
    """A (action-driven) + passive static B, on a temporally-COHERENT video background."""
