"""NoisyShape-Motion: NoisyShape with a predictable (linearly moving) position.

Same colour / shape+size action logic as :class:`shapeenv.NoisyShapeEnv`.
The object moves at constant speed (0.03 of the image width per step) in a
direction drawn once per episode and bounces elastically off a 0.05 margin,
so its position is dynamically predictable. Backgrounds are drawn i.i.d. per
step from a fixed pool of ``num_backgrounds`` (default 20) pre-loaded frames;
the index is exposed as ``info['background_id']``.

Note: the pool is the first ``num_backgrounds`` frames in sorted DAVIS order
(i.e. frames of the first sequence), exactly as used for the paper.

Paper: Appendix E.4, "Selective spatial encoding" (Fig. 18, Table 11).
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

# Number of background images in the pool
NUM_BACKGROUNDS = 20

# Motion parameters
SPEED = 0.03  # normalized units per step (~2 pixels in 64x64)


class NoisyShapeMotionEnv(gym.Env):
    """
    Shape environment with inertial linear motion + bounce and a small pool of DAVIS backgrounds.

    Actions (Discrete(4)):
        0: color_count -= 1  (cycle color backward)
        1: color_count += 1  (cycle color forward)
        2: shape_count -= 1  (cycle shape+size backward)
        3: shape_count += 1  (cycle shape+size forward)

    Shape+size combined: shape_count % 12 determines both.
    Position: uniform linear motion, elastic bounce off edges.
    Background: random selection from 20 pre-loaded DAVIS frames.
    Reward: random noise
    Info: ground truth labels (color_id, shape_id, size_id, position_x, position_y, background_id)
    """
    metadata = {}

    def __init__(self, size=(64, 64), seed=0,
                 davis_path=None,
                 num_backgrounds=NUM_BACKGROUNDS,
                 speed=SPEED,
                 include_attr_obs=False):
        super().__init__()
        davis_path = resolve_davis_path(davis_path)
        self._size = tuple(size)
        self._rng = np.random.RandomState(seed)
        self._include_attr_obs = include_attr_obs
        self._num_backgrounds = num_backgrounds
        self._speed = speed

        # State variables
        self._color_count = 0
        self._shape_count = 0
        self._pos_x = 0.5
        self._pos_y = 0.5
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._background_id = 0
        self._is_first = True
        self._step_count = 0

        # Size radii relative to image size
        self._size_fractions = [0.10, 0.18, 0.26]

        # Margin: keep shape center away from the very edge to avoid clipping
        self._margin = 0.05

        # Pre-load and resize the first N DAVIS frames
        self._bg_images = []
        if os.path.isdir(davis_path):
            all_frame_paths = []
            video_dirs = sorted(glob.glob(os.path.join(davis_path, "*")))
            for vdir in video_dirs:
                if os.path.isdir(vdir):
                    frames = sorted(glob.glob(os.path.join(vdir, "*.jpg")))
                    all_frame_paths.extend(frames)
            
            w, h = self._size
            for path in all_frame_paths[:num_backgrounds]:
                bg_img = Image.open(path).convert('RGB')
                bg_img = bg_img.resize((w, h), Image.BILINEAR)
                self._bg_images.append(np.array(bg_img, dtype=np.uint8))

        if len(self._bg_images) == 0:
            print(f"WARNING: No DAVIS frames found at {davis_path}, using random noise")
        
        print(f"NoisyShapeMotionEnv: pre-loaded {len(self._bg_images)} background images "
              f"(pool size={num_backgrounds}, speed={speed})")

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

    def _get_random_background(self):
        """Get a random background from the pre-loaded pool, return (PIL Image, bg_id)."""
        w, h = self._size
        if len(self._bg_images) > 0:
            idx = self._rng.randint(0, len(self._bg_images))
            return Image.fromarray(self._bg_images[idx]), idx
        else:
            noise = self._rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
            return Image.fromarray(noise), -1

    def _update_position(self):
        """Update position with velocity, bounce off edges."""
        margin = self._margin
        
        # Move
        self._pos_x += self._vel_x
        self._pos_y += self._vel_y
        
        # Bounce off left/right edges
        if self._pos_x < margin:
            self._pos_x = margin + (margin - self._pos_x)
            self._vel_x = -self._vel_x
        elif self._pos_x > 1.0 - margin:
            self._pos_x = (1.0 - margin) - (self._pos_x - (1.0 - margin))
            self._vel_x = -self._vel_x
        
        # Bounce off top/bottom edges
        if self._pos_y < margin:
            self._pos_y = margin + (margin - self._pos_y)
            self._vel_y = -self._vel_y
        elif self._pos_y > 1.0 - margin:
            self._pos_y = (1.0 - margin) - (self._pos_y - (1.0 - margin))
            self._vel_y = -self._vel_y
        
        # Clamp for safety
        self._pos_x = np.clip(self._pos_x, margin, 1.0 - margin)
        self._pos_y = np.clip(self._pos_y, margin, 1.0 - margin)

    def _render_image(self):
        """Render shape at current position on top of random background."""
        w, h = self._size
        img, bg_id = self._get_random_background()
        self._background_id = bg_id
        draw = ImageDraw.Draw(img)

        color = COLORS[self._color_id]
        cx = self._pos_x * w
        cy = self._pos_y * h
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
            'position_x': self._pos_x,
            'position_y': self._pos_y,
            'background_id': self._background_id,
        }

    def _random_velocity(self):
        """Generate a random velocity with fixed speed."""
        angle = self._rng.uniform(0, 2 * math.pi)
        self._vel_x = self._speed * math.cos(angle)
        self._vel_y = self._speed * math.sin(angle)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._color_count = self._rng.randint(0, len(COLORS))
        self._shape_count = self._rng.randint(0, NUM_SHAPE_SIZE)
        
        # Random initial position (within margin)
        margin = self._margin
        self._pos_x = self._rng.uniform(margin, 1.0 - margin)
        self._pos_y = self._rng.uniform(margin, 1.0 - margin)
        
        # Random initial velocity direction, fixed speed
        self._random_velocity()
        
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

        # Update position with inertial motion + bounce
        self._update_position()

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
