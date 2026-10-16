"""shapeenv: the Shape / NoisyShape diagnostic environments from the StarWM paper.

All environments render 64x64 RGB observations (configurable via ``size``) of
one or two geometric shapes whose colour / shape / size are driven by discrete
actions, while objective-irrelevant factors (position, video background) are
re-sampled. Ground-truth labels are returned in ``info`` so learned latents
can be probed.

API (Dreamer-style, as used by the StarWM code base):

    env = shapeenv.make("noisyshape", seed=0, davis_path=".../JPEGImages/480p")
    obs = env.reset()                         # dict: image, is_first, is_terminal
    obs, reward, done, truncated, info = env.step(action)   # action: int

``reset`` returns only the observation. Labels of the current state are
available at any time via ``shapeenv.state_labels(env)``.
"""
from .shape import ShapeEnv
from .shape_fixed import ShapeFixedEnv
from .noisy_shape import NoisyShapeEnv
from .noisy_shape_seq import NoisyShapeSeqEnv
from .noisy_shape_motion import NoisyShapeMotionEnv
from .noisy_two_shape import NoisyTwoShapeEnv
from .passive_shape import PassiveShapeEnv, PassiveShapeSeqEnv
from .textured_shape import TexturedNoisyShapeEnv

__version__ = "1.0.0"

# Registry names contain no underscore so they can be used as the
# ``<suite>`` part of a ``<suite>_<task>`` string in Dreamer-style configs.
ENVS = {
    "shape": ShapeEnv,                      # clean, random position
    "shapefixed": ShapeFixedEnv,            # clean, fixed position
    "noisyshape": NoisyShapeEnv,            # random DAVIS frame per step
    "noisyshapeseq": NoisyShapeSeqEnv,      # sequential DAVIS video
    "noisyshapemotion": NoisyShapeMotionEnv,  # predictable linear motion
    "noisytwoshape": NoisyTwoShapeEnv,      # two shapes, 8 actions
    "passiveshape": PassiveShapeEnv,        # + static passive object, random bg
    "passiveshapeseq": PassiveShapeSeqEnv,  # + static passive object, video bg
    "texnoisyshape": TexturedNoisyShapeEnv,  # textured objects, level=1/2/3
}

# Environments that draw DAVIS backgrounds and accept ``davis_path``.
USES_DAVIS = {
    "noisyshape", "noisyshapeseq", "noisyshapemotion", "noisytwoshape",
    "passiveshape", "passiveshapeseq", "texnoisyshape",
}


def make(name, **kwargs):
    """Instantiate an environment by registry name (see ``shapeenv.ENVS``)."""
    try:
        cls = ENVS[name]
    except KeyError:
        raise ValueError(f"Unknown shapeenv '{name}'. Available: {sorted(ENVS)}") from None
    return cls(**kwargs)


def state_labels(env):
    """Ground-truth labels of the environment's current state.

    Same keys as the ``info`` dict returned by ``step`` (without ``discount``);
    for :class:`NoisyShapeMotionEnv` the velocity is added as ``velocity_x`` /
    ``velocity_y``. Reading labels never touches the environment's RNG.
    """
    labels = dict(env._make_info())
    if hasattr(env, "_vel_x"):
        labels["velocity_x"] = env._vel_x
        labels["velocity_y"] = env._vel_y
    return labels


__all__ = [
    "ShapeEnv", "ShapeFixedEnv", "NoisyShapeEnv", "NoisyShapeSeqEnv",
    "NoisyShapeMotionEnv", "NoisyTwoShapeEnv", "PassiveShapeEnv",
    "PassiveShapeSeqEnv", "TexturedNoisyShapeEnv", "ENVS", "USES_DAVIS",
    "make", "state_labels",
]
