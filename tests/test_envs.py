"""Smoke, determinism and label-consistency tests.

DAVIS is optional here: without ``DAVIS_DIR`` the noisy environments fall back
to random-noise backgrounds, which exercises the same code paths.
"""
import numpy as np
import pytest

import shapeenv
from shapeenv.labels import rollout_labels, rollout_labels_motion, rollout_labels_two_shape


@pytest.mark.parametrize("name", sorted(shapeenv.ENVS))
def test_reset_step_and_determinism(name):
    kw = {"level": 3} if name == "texnoisyshape" else {}
    runs = []
    for _ in range(2):
        env = shapeenv.make(name, seed=3, **kw)
        obs = env.reset()
        assert obs["image"].shape == (64, 64, 3) and obs["image"].dtype == np.uint8
        assert obs["is_first"]
        frames = [obs["image"]]
        rng = np.random.RandomState(0)
        for _ in range(20):
            obs, reward, done, truncated, info = env.step(rng.randint(env.action_space.n))
            assert not obs["is_first"] and not done and not truncated
            assert isinstance(reward, float)
            frames.append(obs["image"])
        runs.append(np.stack(frames))
    np.testing.assert_array_equal(runs[0], runs[1])


def _rollout_vs_env(name, rollout, keymap, extra=None, horizon=15):
    env = shapeenv.make(name, seed=11)
    env.reset()
    start = shapeenv.state_labels(env)
    start_arr = {k: np.array([start[v]]) for k, v in keymap.items()}
    if extra:
        start_arr.update({k: np.array([start[v]]) for k, v in extra.items()})
    rng = np.random.RandomState(1)
    actions = rng.randint(env.action_space.n, size=(horizon, 1))
    pred = rollout(start_arr, actions, horizon)
    for h in range(horizon):
        env.step(int(actions[h, 0]))
        truth = shapeenv.state_labels(env)
        for k, v in keymap.items():
            assert pred[h + 1][k][0] == truth[v], (name, h, k)
    return env, pred


def test_labels_noisyshape():
    _rollout_vs_env("noisyshape", rollout_labels,
                    {"color": "color_id", "shape": "shape_id", "size": "size_id"})


def test_labels_two_shape():
    _rollout_vs_env("noisytwoshape", rollout_labels_two_shape,
                    {"s1_color": "s1_color_id", "s2_color": "s2_color_id",
                     "s1_shape": "s1_shape_id", "s2_shape": "s2_shape_id"})


def test_labels_motion_position():
    env = shapeenv.make("noisyshapemotion", seed=5)
    env.reset()
    s = shapeenv.state_labels(env)
    start = {"color": np.array([s["color_id"]]), "shape": np.array([s["shape_id"]]),
             "size": np.array([s["size_id"]]), "pos_x": np.array([s["position_x"]]),
             "pos_y": np.array([s["position_y"]]), "vel_x": np.array([s["velocity_x"]]),
             "vel_y": np.array([s["velocity_y"]])}
    actions = np.random.RandomState(2).randint(4, size=(30, 1))
    pred = rollout_labels_motion(start, actions, 30)
    for h in range(30):
        env.step(int(actions[h, 0]))
        t = shapeenv.state_labels(env)
        assert pred[h + 1]["size"][0] == t["size_id"]
        assert abs(pred[h + 1]["pos_x"][0] - t["position_x"]) < 1e-9
        assert abs(pred[h + 1]["pos_y"][0] - t["position_y"]) < 1e-9


def test_chance_level():
    assert abs(shapeenv.labels.position_chance_mse() - 0.0711) < 5e-4


def test_collect_shapes():
    from shapeenv.collect import collect_trajectories
    imgs, acts, isf, labels = collect_trajectories("shape", 2, episode_length=10)
    assert imgs.shape == (2, 11, 64, 64, 3) and acts.shape == (2, 10, 4)
    assert isf[:, 0].all() and not isf[:, 1:].any()
    assert labels["color_id"].shape == (2, 11)
