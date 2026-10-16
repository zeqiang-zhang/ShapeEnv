"""Shape-Fixed: the Shape environment with the shape always at the image centre.

Same actions, colours, sizes and noise reward as :class:`shapeenv.ShapeEnv`,
but the position never changes, so colour/shape/size are the only sources of
variation.

Paper: "Shape Env-Fixed", Appendix C.1 / Table 12.
"""

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw
import math


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


class ShapeFixedEnv(gym.Env):
    """
    Shape environment with fixed center position.

    Actions (Discrete(4)):
        0: color_count -= 1  (cycle color backward)
        1: color_count += 1  (cycle color forward)
        2: shape_count -= 1  (cycle shape+size backward)
        3: shape_count += 1  (cycle shape+size forward)

    Shape and size are combined: shape_count % 12 determines both.
      shape_id = (shape_count % 12) // 3
      size_id  = (shape_count % 12) % 3

    Position is ALWAYS at the center (0.5, 0.5) - never randomized.

    Observations: Dict with 'image' (H x W x 3, uint8), 'is_first', 'is_terminal'
    Reward: random noise
    Info: ground truth labels for analysis
    """
    metadata = {}

    def __init__(self, size=(64, 64), seed=0, include_attr_obs=False):
        super().__init__()
        self._size = tuple(size)
        self._rng = np.random.RandomState(seed)
        self._include_attr_obs = include_attr_obs

        # State variables (using counters like original ShapeEnv.py)
        self._color_count = 0
        self._shape_count = 0
        self._pos = (0.5, 0.5)  # ALWAYS center
        self._is_first = True
        self._step_count = 0

        # Size radii relative to image size (fraction of min dimension)
        self._size_fractions = [0.10, 0.18, 0.26]  # small, medium, large

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
        """Get the radius in pixels for the current size."""
        min_dim = min(self._size)
        return self._size_fractions[self._size_id] * min_dim

    def _render_image(self):
        """Render the current state as an RGB image using PIL."""
        w, h = self._size
        img = Image.new('RGB', (w, h), (255, 255, 255))  # white background
        draw = ImageDraw.Draw(img)

        color = COLORS[self._color_id]
        cx = self._pos[0] * w
        cy = self._pos[1] * h
        r = self._get_radius_pixels()

        if self._shape_id == 0:  # circle
            bbox = [cx - r, cy - r, cx + r, cy + r]
            draw.ellipse(bbox, fill=color)

        elif self._shape_id == 1:  # equilateral triangle (pointing up)
            pts = []
            for i in range(3):
                angle = math.pi / 2 + 2 * math.pi * i / 3
                px = cx + r * math.cos(angle)
                py = cy - r * math.sin(angle)  # y is inverted in image coords
                pts.append((px, py))
            draw.polygon(pts, fill=color)

        elif self._shape_id == 2:  # square
            half = r * math.sqrt(2) / 2  # inscribed in circle of radius r
            bbox = [cx - half, cy - half, cx + half, cy + half]
            draw.rectangle(bbox, fill=color)

        elif self._shape_id == 3:  # pentagon
            pts = []
            for ux, uy in _PENTAGON_UNIT:
                px = cx + r * ux
                py = cy - r * uy  # y inverted
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
        self._pos = (0.5, 0.5)  # Always center
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

        # Position is NOT randomized—stays at center
        # (self._pos remains (0.5, 0.5))

        self._step_count += 1
        reward = float(self._rng.randn())  # random reward
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
