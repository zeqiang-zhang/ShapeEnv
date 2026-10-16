"""Textured NoisyShape: NoisyShape with textured instead of solid-colour objects.

The colour attribute is replaced by a texture ID (4 textures, still cycled by
actions 0/1), giving a three-level foreground-complexity ladder while the MDP
stays isomorphic to :class:`shapeenv.NoisyShapeEnv`:

* level 1 (L1): procedural textures from an external domain (checkerboard,
  horizontal/vertical stripes, dot grid), generated in code;
* level 2 (L2): four fixed centre crops of DAVIS frames, i.e. the same visual
  domain as the background;
* level 3 (L3): the L2 textures alpha-blended into the background
  (``alpha=0.5``) inside a feathered (Gaussian-blurred) shape mask.

Textures are anchored to the shape's bounding box, so appearance does not leak
absolute position. ``info['texture_id']`` holds the label (``info['color_id']``
is kept as an alias so colour probes work unchanged).

Paper: Appendix F.3, "Texture, low contrast and photorealistic scenes" (Table 13).
"""

import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .noisy_shape import (
    NoisyShapeEnv,
    _PENTAGON_UNIT,
    NUM_SHAPE_SIZE,
)
import math


NUM_TEXTURES = 4  # replaces the 4 colors of NoisyShapeEnv


class TexturedNoisyShapeEnv(NoisyShapeEnv):
    """NoisyShapeEnv with texture-ID instead of color, at 3 complexity levels."""

    def __init__(self, size=(64, 64), seed=0,
                 davis_path=None,
                 level=1, alpha=0.5, edge_blur=1.0, include_attr_obs=False):
        super().__init__(size=size, seed=seed, davis_path=davis_path,
                         include_attr_obs=include_attr_obs)
        assert level in (1, 2, 3), f"level must be 1/2/3, got {level}"
        self._level = int(level)
        self._alpha = float(alpha)
        self._edge_blur = float(edge_blur)
        # Deterministic texture bank (fixed appearance per texture-ID).
        self._textures = self._build_textures()
        print(f"TexturedNoisyShapeEnv: level={self._level} alpha={self._alpha} "
              f"edge_blur={self._edge_blur}; {NUM_TEXTURES} textures built.")

    # ------------------------------------------------------------------
    # texture bank
    # ------------------------------------------------------------------
    @property
    def _texture_id(self):
        # reuse the color counter as the texture counter (identical dynamics)
        return self._color_count % NUM_TEXTURES

    def _build_textures(self):
        w, h = self._size
        if self._level == 1:
            return self._build_procedural_textures(h, w)
        # L2 / L3 use crops from the SAME DAVIS pool used as background.
        return self._build_davis_textures(h, w)

    @staticmethod
    def _build_procedural_textures(h, w):
        """4 distinct synthetic textures (different domain from natural video)."""
        yy, xx = np.mgrid[0:h, 0:w]
        texs = []
        # 0: checkerboard (8px) light/dark grey
        chk = (((xx // 8) + (yy // 8)) % 2).astype(np.uint8)
        t0 = np.where(chk[..., None] == 1,
                      np.array([210, 210, 210], np.uint8),
                      np.array([45, 45, 45], np.uint8))
        texs.append(t0.astype(np.uint8))
        # 1: horizontal stripes (4px), warm two-tone
        hs = ((yy // 4) % 2).astype(np.uint8)
        t1 = np.where(hs[..., None] == 1,
                      np.array([235, 130, 40], np.uint8),
                      np.array([60, 30, 10], np.uint8))
        texs.append(t1.astype(np.uint8))
        # 2: vertical stripes (4px), cool two-tone
        vs = ((xx // 4) % 2).astype(np.uint8)
        t2 = np.where(vs[..., None] == 1,
                      np.array([40, 130, 235], np.uint8),
                      np.array([10, 30, 60], np.uint8))
        texs.append(t2.astype(np.uint8))
        # 3: dot grid (radius 3, spacing 10)
        cx = xx % 10 - 5
        cy = yy % 10 - 5
        dot = (cx * cx + cy * cy <= 9)
        t3 = np.where(dot[..., None],
                      np.array([60, 200, 120], np.uint8),
                      np.array([230, 230, 230], np.uint8))
        texs.append(t3.astype(np.uint8))
        return texs

    def _build_davis_textures(self, h, w):
        """4 fixed square crops from the DAVIS background pool (same domain)."""
        if len(self._bg_paths) == 0:
            print("WARNING: no DAVIS frames for textures; falling back to procedural")
            return self._build_procedural_textures(h, w)
        n = len(self._bg_paths)
        # deterministic, well-spread frame indices (independent of episode RNG)
        idxs = [int((k + 0.5) * n / NUM_TEXTURES) % n for k in range(NUM_TEXTURES)]
        texs = []
        for i in idxs:
            img = Image.open(self._bg_paths[i]).convert("RGB")
            iw, ih = img.size
            s = min(iw, ih)
            left, top = (iw - s) // 2, (ih - s) // 2
            img = img.crop((left, top, left + s, top + s)).resize((w, h), Image.BILINEAR)
            texs.append(np.array(img, dtype=np.uint8))
        return texs

    # ------------------------------------------------------------------
    # shape mask (white shape on black), reusing NoisyShape geometry
    # ------------------------------------------------------------------
    def _shape_mask(self, cx, cy, r):
        w, h = self._size
        m = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(m)
        sid = self._shape_id
        if sid == 0:  # circle
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        elif sid == 1:  # triangle
            pts = []
            for i in range(3):
                a = math.pi / 2 + 2 * math.pi * i / 3
                pts.append((cx + r * math.cos(a), cy - r * math.sin(a)))
            d.polygon(pts, fill=255)
        elif sid == 2:  # square
            half = r * math.sqrt(2) / 2
            d.rectangle([cx - half, cy - half, cx + half, cy + half], fill=255)
        elif sid == 3:  # pentagon
            pts = [(cx + r * ux, cy - r * uy) for ux, uy in _PENTAGON_UNIT]
            d.polygon(pts, fill=255)
        return m  # PIL 'L' image

    def _render_image(self):
        w, h = self._size
        bg = np.array(self._get_random_background(), dtype=np.float32)  # (h,w,3)

        cx = self._pos[0] * w
        cy = self._pos[1] * h
        r = self._get_radius_pixels()

        # texture layer: resize this texture-ID to the shape bbox, paste anchored
        # to the shape (position-invariant appearance), else background elsewhere.
        tex_src = self._textures[self._texture_id]
        bw = max(1, int(round(2 * r)))
        patch = np.array(
            Image.fromarray(tex_src).resize((bw, bw), Image.BILINEAR),
            dtype=np.float32,
        )
        tex_full = bg.copy()
        x0, y0 = int(round(cx - r)), int(round(cy - r))
        # clip paste region to image bounds
        px0, py0 = max(0, x0), max(0, y0)
        px1, py1 = min(w, x0 + bw), min(h, y0 + bw)
        if px1 > px0 and py1 > py0:
            tex_full[py0:py1, px0:px1, :] = patch[py0 - y0:py1 - y0, px0 - x0:px1 - x0, :]

        mask = np.array(self._shape_mask(cx, cy, r), dtype=np.float32) / 255.0  # (h,w)

        if self._level in (1, 2):
            m = mask[..., None]
            out = m * tex_full + (1.0 - m) * bg
        else:  # level 3: alpha blend inside a feathered mask
            if self._edge_blur > 0:
                soft = np.array(
                    self._shape_mask(cx, cy, r).filter(
                        ImageFilter.GaussianBlur(self._edge_blur)),
                    dtype=np.float32,
                ) / 255.0
            else:
                soft = mask
            s = soft[..., None]
            blended = self._alpha * tex_full + (1.0 - self._alpha) * bg
            out = s * blended + (1.0 - s) * bg

        return np.clip(out, 0, 255).astype(np.uint8)

    def _make_info(self):
        info = super()._make_info()
        # texture-ID is the probe target that replaces color; keep `color_id` as an
        # alias so the existing probe (which reads info['color_id']) works unchanged.
        info["texture_id"] = self._color_id
        info["level"] = self._level
        return info
