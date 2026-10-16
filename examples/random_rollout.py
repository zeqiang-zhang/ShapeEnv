"""Minimal usage: roll out a random policy and read the ground-truth labels.

    python examples/random_rollout.py --env noisyshape --davis_path /path/to/DAVIS/JPEGImages/480p
"""
import argparse
import os

import numpy as np

import shapeenv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="noisyshape", choices=sorted(shapeenv.ENVS))
    ap.add_argument("--davis_path", default=os.environ.get("DAVIS_DIR"))
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()

    kwargs = {"davis_path": args.davis_path} if args.env in shapeenv.USES_DAVIS else {}
    env = shapeenv.make(args.env, seed=0, **kwargs)
    obs = env.reset()
    print("obs keys:", sorted(obs), "image:", obs["image"].shape, obs["image"].dtype)
    print("t=0 labels:", shapeenv.state_labels(env))
    rng = np.random.RandomState(0)
    for t in range(1, args.steps + 1):
        action = rng.randint(env.action_space.n)
        obs, reward, done, truncated, info = env.step(action)
        print(f"t={t} action={action} reward={reward:+.2f} labels={ {k: v for k, v in info.items() if k != 'discount'} }")


if __name__ == "__main__":
    main()
