"""Ground-truth labels for open-loop imagination (Appendix C.2 of the paper).

Given the labels of a starting state and a sequence of (imagined) actions,
these functions roll the environment's deterministic attribute logic forward,
so that a probe applied to an imagined latent at horizon ``h`` can be scored
against the correct label. All inputs are flat numpy arrays over samples;
``imagined_actions`` has shape ``(H, N)`` with integer actions.

The arithmetic is copied verbatim from the probe scripts used for the paper.
"""
import numpy as np

NUM_COLORS = 4
NUM_SHAPES = 4
NUM_SIZES = 3
NUM_SHAPE_SIZE = NUM_SHAPES * NUM_SIZES  # 12

# NoisyShapeMotionEnv constants (must match noisy_shape_motion.py)
MOTION_MARGIN = 0.05


def rollout_labels(start_labels, imagined_actions, horizon):
    """Shape / NoisyShape family (4 actions: 0/1 colour -/+, 2/3 shape+size -/+).

    ``start_labels``: dict with integer arrays ``color``, ``shape``, ``size``.
    Returns ``{h: {'color', 'shape', 'size'}}`` for ``h = 0..horizon``.
    Also valid for the textured (texture id in place of colour) and passive
    (object A) variants, whose action logic is identical.
    """
    result = {0: {'color': start_labels['color'].copy(), 'shape': start_labels['shape'].copy(), 'size': start_labels['size'].copy()}}
    cc = start_labels['color'].copy().astype(np.int64)
    sc = start_labels['shape'].copy().astype(np.int64)*3 + start_labels['size'].copy().astype(np.int64)
    for h in range(horizon):
        a = imagined_actions[h]
        cc[a==0]-=1; cc[a==1]+=1; sc[a==2]-=1; sc[a==3]+=1
        result[h+1] = {'color':(cc%4).astype(np.int64), 'shape':((sc%12)//3).astype(np.int64), 'size':((sc%12)%3).astype(np.int64)}
    return result


def rollout_labels_motion(start_labels, imagined_actions, horizon, margin=MOTION_MARGIN):
    """NoisyShapeMotionEnv: attribute logic as above plus action-independent
    linear motion with elastic bounce.

    ``start_labels`` additionally needs float arrays ``pos_x``, ``pos_y``,
    ``vel_x``, ``vel_y``. Returns ``{h: {'color','shape','size','pos_x','pos_y'}}``.
    """
    MARGIN = margin
    result = {}

    # Color / shape / size
    cc = start_labels['color'].copy().astype(np.int64)
    sc = start_labels['shape'].copy().astype(np.int64) * 3 + start_labels['size'].copy().astype(np.int64)
    result[0] = {
        'color': cc % 4,
        'shape': (sc % 12) // 3,
        'size': (sc % 12) % 3,
        'pos_x': start_labels['pos_x'].copy(),
        'pos_y': start_labels['pos_y'].copy(),
    }

    # Position state
    px = start_labels['pos_x'].copy().astype(np.float64)
    py = start_labels['pos_y'].copy().astype(np.float64)
    vx = start_labels['vel_x'].copy().astype(np.float64)
    vy = start_labels['vel_y'].copy().astype(np.float64)

    for h in range(horizon):
        a = imagined_actions[h]
        # Color/shape update
        cc[a == 0] -= 1
        cc[a == 1] += 1
        sc[a == 2] -= 1
        sc[a == 3] += 1

        # Position update (bounce)
        px += vx
        py += vy
        # Bounce x
        lo = px < MARGIN
        px[lo] = MARGIN + (MARGIN - px[lo])
        vx[lo] = -vx[lo]
        hi = px > 1.0 - MARGIN
        px[hi] = (1.0 - MARGIN) - (px[hi] - (1.0 - MARGIN))
        vx[hi] = -vx[hi]
        # Bounce y
        lo = py < MARGIN
        py[lo] = MARGIN + (MARGIN - py[lo])
        vy[lo] = -vy[lo]
        hi = py > 1.0 - MARGIN
        py[hi] = (1.0 - MARGIN) - (py[hi] - (1.0 - MARGIN))
        vy[hi] = -vy[hi]
        # Clamp
        px = np.clip(px, MARGIN, 1.0 - MARGIN)
        py = np.clip(py, MARGIN, 1.0 - MARGIN)

        result[h + 1] = {
            'color': (cc % 4).astype(np.int64),
            'shape': ((sc % 12) // 3).astype(np.int64),
            'size': ((sc % 12) % 3).astype(np.int64),
            'pos_x': px.copy(),
            'pos_y': py.copy(),
        }
    return result


def rollout_labels_two_shape(start_labels, imagined_actions, horizon):
    """NoisyTwoShapeEnv (8 actions, see noisy_two_shape.py).

    ``start_labels`` keys: ``s1_color``, ``s2_color``, ``s1_shape``, ``s2_shape``.
    Returns ``{h: {key: array}}`` for ``h = 0..horizon``.
    """
    out = {0: {k: start_labels[k].copy() for k in
               ["s1_color", "s2_color", "s1_shape", "s2_shape"]}}
    s1c = start_labels["s1_color"].copy().astype(np.int64)
    s2c = start_labels["s2_color"].copy().astype(np.int64)
    s1s = start_labels["s1_shape"].copy().astype(np.int64)
    s2s = start_labels["s2_shape"].copy().astype(np.int64)
    for h in range(horizon):
        a = imagined_actions[h]
        s1c[a == 0] -= 1; s1c[a == 1] += 1
        s2c[a == 2] -= 1; s2c[a == 3] += 1
        s1s[a == 4] -= 1; s1s[a == 5] += 1
        s2s[a == 6] -= 1; s2s[a == 7] += 1
        out[h + 1] = {
            "s1_color": (s1c % NUM_COLORS).astype(np.int64),
            "s2_color": (s2c % NUM_COLORS).astype(np.int64),
            "s1_shape": (s1s % NUM_SHAPES).astype(np.int64),
            "s2_shape": (s2s % NUM_SHAPES).astype(np.int64),
        }
    return out


def position_chance_mse(size_fractions=(0.10, 0.18, 0.26)):
    """Chance-level position MSE (sum over x and y) of the constant predictor
    0.5 for envs that re-sample position uniformly in [m, 1-m]^2 with
    m = radius fraction, averaged over the three sizes (Appendix C.1).
    Returns ~0.0711 for the default sizes."""
    return float(np.mean([(1 - 2 * m) ** 2 / 6 for m in size_fractions]))
