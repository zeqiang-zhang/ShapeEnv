"""NoisyTwoShape: two independently controlled shapes on random DAVIS frames.

Shape 1 is large (radius 0.26 of the image side), shape 2 is small (0.10).
Each has its own colour (4 values) and shape (4 values) counter; sizes are
fixed. Actions (Discrete(8)):
    0/1 shape-1 colour -/+,  2/3 shape-2 colour -/+,
    4/5 shape-1 shape  -/+,  6/7 shape-2 shape  -/+.
Both positions are re-sampled every step under a non-overlap constraint and
the background is a random DAVIS frame per step. Reward is i.i.d. Gaussian
noise. Labels are exposed as ``info['s{1,2}_{color_id,shape_id,position_x,
position_y}']``.

Paper: Appendix F.2, "Multi-object extension: NoisyTwoShape" (Figs. 21-22).
"""

import os
import glob
import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw
import math

from ._davis import resolve_davis_path


# Pre-computed pentagon unit vertices (pointing up)
_PENTAGON_UNIT = []
for i in range(5):
    angle = math.pi / 2 + 2 * math.pi * i / 5
    _PENTAGON_UNIT.append((math.cos(angle), math.sin(angle)))
_PENTAGON_UNIT = np.array(_PENTAGON_UNIT, dtype=np.float64)

COLORS = [
    (255, 0, 0),      # red
    (0, 180, 0),      # green
    (0, 0, 255),      # blue
    (0, 0, 0),        # black
]
COLOR_NAMES = ['red', 'green', 'blue', 'black']
SHAPE_NAMES = ['circle', 'triangle', 'square', 'pentagon']
NUM_SHAPES = len(SHAPE_NAMES)
NUM_COLORS = len(COLORS)

SIZE_FRAC_LARGE = 0.26
SIZE_FRAC_SMALL = 0.10


class NoisyTwoShapeEnv(gym.Env):
    """Two shapes with random DAVIS video frames as distracting background.

    Action space, color/shape counters, and per-shape sizes match
    :class:`TwoShapeEnv`. The only difference is the background: instead of a
    plain white canvas, every step samples a random DAVIS frame and draws the
    two shapes on top.
    """
    metadata = {}

    def __init__(self, size=(64, 64), seed=0,
                 davis_path=None,
                 include_attr_obs=False):
        super().__init__()
        davis_path = resolve_davis_path(davis_path)
        self._size = tuple(size)
        self._rng = np.random.RandomState(seed)
        self._davis_path = davis_path
        self._include_attr_obs = include_attr_obs

        # Per-shape counters
        self._s1_color_count = 0
        self._s1_shape_count = 0
        self._s2_color_count = 0
        self._s2_shape_count = 0

        # Per-shape positions
        self._s1_pos = None
        self._s2_pos = None

        self._is_first = True
        self._step_count = 0

        # Load all DAVIS frame paths
        self._bg_paths = []
        if os.path.isdir(davis_path):
            video_dirs = sorted(glob.glob(os.path.join(davis_path, "*")))
            for vdir in video_dirs:
                if os.path.isdir(vdir):
                    frames = sorted(glob.glob(os.path.join(vdir, "*.jpg")))
                    self._bg_paths.extend(frames)
        if len(self._bg_paths) == 0:
            print(f"WARNING: No DAVIS frames found at {davis_path}, using random noise")

        print(f"NoisyTwoShapeEnv: loaded {len(self._bg_paths)} background frames")

    # --- attribute id getters ---------------------------------------------
    @property
    def _s1_color_id(self):
        return self._s1_color_count % NUM_COLORS

    @property
    def _s2_color_id(self):
        return self._s2_color_count % NUM_COLORS

    @property
    def _s1_shape_id(self):
        return self._s1_shape_count % NUM_SHAPES

    @property
    def _s2_shape_id(self):
        return self._s2_shape_count % NUM_SHAPES

    @property
    def observation_space(self):
        spaces = {
            'image': gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
        }
        if self._include_attr_obs:
            spaces['attr_s1_color'] = gym.spaces.Box(0, NUM_COLORS - 1, shape=(), dtype=np.int64)
            spaces['attr_s2_color'] = gym.spaces.Box(0, NUM_COLORS - 1, shape=(), dtype=np.int64)
            spaces['attr_s1_shape'] = gym.spaces.Box(0, NUM_SHAPES - 1, shape=(), dtype=np.int64)
            spaces['attr_s2_shape'] = gym.spaces.Box(0, NUM_SHAPES - 1, shape=(), dtype=np.int64)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return gym.spaces.Discrete(8)

    # --- geometry helpers --------------------------------------------------
    def _radius_pixels(self, size_frac):
        return size_frac * min(self._size)

    def _sample_position(self, size_frac):
        r = self._radius_pixels(size_frac)
        margin = min(r / min(self._size), 0.45)
        x = self._rng.uniform(margin, 1.0 - margin)
        y = self._rng.uniform(margin, 1.0 - margin)
        return (x, y)

    def _sample_two_positions(self):
        r1 = self._radius_pixels(SIZE_FRAC_LARGE)
        r2 = self._radius_pixels(SIZE_FRAC_SMALL)
        gap = 2.0
        min_dist_norm = (r1 + r2 + gap) / min(self._size)
        p1 = self._sample_position(SIZE_FRAC_LARGE)
        for _ in range(200):
            p2 = self._sample_position(SIZE_FRAC_SMALL)
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]
            if math.hypot(dx, dy) >= min_dist_norm:
                return p1, p2
        return p1, p2

    # --- background --------------------------------------------------------
    def _get_random_background(self):
        w, h = self._size
        if len(self._bg_paths) > 0:
            idx = self._rng.randint(0, len(self._bg_paths))
            bg_img = Image.open(self._bg_paths[idx]).convert('RGB')
            bg_img = bg_img.resize((w, h), Image.BILINEAR)
            return bg_img
        else:
            noise = self._rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
            return Image.fromarray(noise)

    # --- rendering ---------------------------------------------------------
    def _draw_shape(self, draw, shape_id, color, cx, cy, r):
        if shape_id == 0:  # circle
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        elif shape_id == 1:  # equilateral triangle (pointing up)
            pts = []
            for i in range(3):
                angle = math.pi / 2 + 2 * math.pi * i / 3
                px = cx + r * math.cos(angle)
                py = cy - r * math.sin(angle)
                pts.append((px, py))
            draw.polygon(pts, fill=color)
        elif shape_id == 2:  # square
            half = r * math.sqrt(2) / 2
            draw.rectangle([cx - half, cy - half, cx + half, cy + half], fill=color)
        elif shape_id == 3:  # pentagon
            pts = []
            for ux, uy in _PENTAGON_UNIT:
                pts.append((cx + r * ux, cy - r * uy))
            draw.polygon(pts, fill=color)

    def _render_image(self):
        """Render the two shapes on top of a random DAVIS background."""
        w, h = self._size
        img = self._get_random_background()
        draw = ImageDraw.Draw(img)
        # shape_1 (large)
        r1 = self._radius_pixels(SIZE_FRAC_LARGE)
        self._draw_shape(draw, self._s1_shape_id, COLORS[self._s1_color_id],
                         self._s1_pos[0] * w, self._s1_pos[1] * h, r1)
        # shape_2 (small)
        r2 = self._radius_pixels(SIZE_FRAC_SMALL)
        self._draw_shape(draw, self._s2_shape_id, COLORS[self._s2_color_id],
                         self._s2_pos[0] * w, self._s2_pos[1] * h, r2)
        return np.array(img, dtype=np.uint8)

    # --- gym API -----------------------------------------------------------
    def _make_obs(self):
        obs = {
            'image': self._render_image(),
            'is_first': self._is_first,
            'is_terminal': False,
        }
        if self._include_attr_obs:
            obs['attr_s1_color'] = np.int64(self._s1_color_id)
            obs['attr_s2_color'] = np.int64(self._s2_color_id)
            obs['attr_s1_shape'] = np.int64(self._s1_shape_id)
            obs['attr_s2_shape'] = np.int64(self._s2_shape_id)
        return obs

    def _make_info(self):
        return {
            's1_color_id': self._s1_color_id,
            's1_shape_id': self._s1_shape_id,
            's1_position_x': self._s1_pos[0],
            's1_position_y': self._s1_pos[1],
            's2_color_id': self._s2_color_id,
            's2_shape_id': self._s2_shape_id,
            's2_position_x': self._s2_pos[0],
            's2_position_y': self._s2_pos[1],
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._s1_color_count = self._rng.randint(0, NUM_COLORS)
        self._s2_color_count = self._rng.randint(0, NUM_COLORS)
        self._s1_shape_count = self._rng.randint(0, NUM_SHAPES)
        self._s2_shape_count = self._rng.randint(0, NUM_SHAPES)
        self._s1_pos, self._s2_pos = self._sample_two_positions()
        self._is_first = True
        self._step_count = 0
        obs = self._make_obs()
        self._is_first = False
        return obs

    def step(self, action):
        action = int(action)
        if action == 0:
            self._s1_color_count -= 1
        elif action == 1:
            self._s1_color_count += 1
        elif action == 2:
            self._s2_color_count -= 1
        elif action == 3:
            self._s2_color_count += 1
        elif action == 4:
            self._s1_shape_count -= 1
        elif action == 5:
            self._s1_shape_count += 1
        elif action == 6:
            self._s2_shape_count -= 1
        elif action == 7:
            self._s2_shape_count += 1

        self._s1_pos, self._s2_pos = self._sample_two_positions()

        self._step_count += 1
        reward = float(self._rng.randn())
        obs = self._make_obs()
        info = self._make_info()
        info['discount'] = np.array(1.0, dtype=np.float32)
        done = False
        truncated = False
        return obs, reward, done, truncated, info

    def render(self, mode='rgb_array'):
        return self._render_image()

    def close(self):
        pass
