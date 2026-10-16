"""NoisyShape: the Shape environment on random DAVIS video frames.

Same action logic as :class:`shapeenv.ShapeEnv` (0/1 colour, 2/3 shape+size).
Two objective-irrelevant factors are re-sampled independently at every step:
the object position and the background (a random frame from all DAVIS 2017
480p sequences, resized to the observation size). Colour, shape and size are
deterministic given the action sequence. Reward is i.i.d. Gaussian noise.

Paper: the main diagnostic environment (Section 4.1, Appendix C.1).
"""

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw
import math
import os
import glob

from ._davis import resolve_davis_path


# Pre-computed pentagon unit vertices (pointing up)
_PENTAGON_UNIT = []
for i in range(5):
    angle = math.pi / 2 + 2 * math.pi * i / 5
    _PENTAGON_UNIT.append((math.cos(angle), math.sin(angle)))
_PENTAGON_UNIT = np.array(_PENTAGON_UNIT, dtype=np.float64)

# Color definitions: (R, G, B)
COLORS = [
    (255, 0, 0),      # red
    (0, 180, 0),      # green
    (0, 0, 255),      # blue
    (0, 0, 0),        # black
]
COLOR_NAMES = ['red', 'green', 'blue', 'black']
SHAPE_NAMES = ['circle', 'triangle', 'square', 'pentagon']
SIZE_NAMES = ['small', 'medium', 'large']
NUM_SHAPES = len(SHAPE_NAMES)
NUM_SIZES = len(SIZE_NAMES)
NUM_SHAPE_SIZE = NUM_SHAPES * NUM_SIZES  # 12 combinations


class NoisyShapeEnv(gym.Env):
    """
    Shape environment with random DAVIS video frames as distracting background.

    Actions (Discrete(4)):
        0: color_count -= 1  (cycle color backward)
        1: color_count += 1  (cycle color forward)
        2: shape_count -= 1  (cycle shape+size backward)
        3: shape_count += 1  (cycle shape+size forward)

    Shape+size combined: shape_count % 12 determines both.
    Position: ALWAYS randomized every step.
    Background: random DAVIS frame, changes every step.
    Reward: random noise
    Info: ground truth labels for analysis
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

        # State variables
        self._color_count = 0
        self._shape_count = 0
        self._pos = None
        self._is_first = True
        self._step_count = 0

        # Size radii relative to image size
        self._size_fractions = [0.10, 0.18, 0.26]

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

        print(f"NoisyShapeEnv: loaded {len(self._bg_paths)} background frames")

    @property
    def _color_id(self):
        return self._color_count % len(COLORS)

    @property
    def _shape_id(self):
        combo = self._shape_count % NUM_SHAPE_SIZE
        return combo // NUM_SIZES

    @property
    def _size_id(self):
        combo = self._shape_count % NUM_SHAPE_SIZE
        return combo % NUM_SIZES

    @property
    def observation_space(self):
        spaces = {
            'image': gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
        }
        if self._include_attr_obs:
            spaces['attr_color'] = gym.spaces.Box(0, 3, shape=(), dtype=np.int64)
            spaces['attr_shape'] = gym.spaces.Box(0, 3, shape=(), dtype=np.int64)
            spaces['attr_size'] = gym.spaces.Box(0, 2, shape=(), dtype=np.int64)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return gym.spaces.Discrete(4)

    def _get_radius_pixels(self):
        min_dim = min(self._size)
        return self._size_fractions[self._size_id] * min_dim

    def _random_position(self):
        r = self._get_radius_pixels()
        min_dim = min(self._size)
        margin = r / min_dim
        margin = min(margin, 0.45)
        x = self._rng.uniform(margin, 1.0 - margin)
        y = self._rng.uniform(margin, 1.0 - margin)
        return (x, y)

    def _get_random_background(self):
        """Get a random DAVIS frame resized to env size, or random noise."""
        w, h = self._size
        if len(self._bg_paths) > 0:
            idx = self._rng.randint(0, len(self._bg_paths))
            bg_img = Image.open(self._bg_paths[idx]).convert('RGB')
            bg_img = bg_img.resize((w, h), Image.BILINEAR)
            return bg_img
        else:
            noise = self._rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
            return Image.fromarray(noise)

    def _render_image(self):
        """Render shape on top of random DAVIS background."""
        w, h = self._size
        img = self._get_random_background()
        draw = ImageDraw.Draw(img)

        color = COLORS[self._color_id]
        cx = self._pos[0] * w
        cy = self._pos[1] * h
        r = self._get_radius_pixels()

        if self._shape_id == 0:  # circle
            bbox = [cx - r, cy - r, cx + r, cy + r]
            draw.ellipse(bbox, fill=color)

        elif self._shape_id == 1:  # triangle
            pts = []
            for i in range(3):
                angle = math.pi / 2 + 2 * math.pi * i / 3
                px = cx + r * math.cos(angle)
                py = cy - r * math.sin(angle)
                pts.append((px, py))
            draw.polygon(pts, fill=color)

        elif self._shape_id == 2:  # square
            half = r * math.sqrt(2) / 2
            bbox = [cx - half, cy - half, cx + half, cy + half]
            draw.rectangle(bbox, fill=color)

        elif self._shape_id == 3:  # pentagon
            pts = []
            for ux, uy in _PENTAGON_UNIT:
                px = cx + r * ux
                py = cy - r * uy
                pts.append((px, py))
            draw.polygon(pts, fill=color)

        return np.array(img, dtype=np.uint8)

    def _make_obs(self):
        obs = {
            'image': self._render_image(),
            'is_first': self._is_first,
            'is_terminal': False,
        }
        if self._include_attr_obs:
            obs['attr_color'] = np.int64(self._color_id)
            obs['attr_shape'] = np.int64(self._shape_id)
            obs['attr_size'] = np.int64(self._size_id)
        return obs

    def _make_info(self):
        return {
            'color_id': self._color_id,
            'shape_id': self._shape_id,
            'size_id': self._size_id,
            'position_x': self._pos[0],
            'position_y': self._pos[1],
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._color_count = self._rng.randint(0, len(COLORS))
        self._shape_count = self._rng.randint(0, NUM_SHAPE_SIZE)
        self._pos = self._random_position()
        self._is_first = True
        self._step_count = 0
        obs = self._make_obs()
        self._is_first = False
        return obs

    def step(self, action):
        action = int(action)
        if action == 0:    # cycle color backward
            self._color_count -= 1
        elif action == 1:  # cycle color forward
            self._color_count += 1
        elif action == 2:  # cycle shape+size backward
            self._shape_count -= 1
        elif action == 3:  # cycle shape+size forward
            self._shape_count += 1

        # Position is ALWAYS randomized every step
        self._pos = self._random_position()

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
