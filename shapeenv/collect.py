"""Random-policy trajectory collection with ground-truth labels.

Used to build the probe datasets of the paper (Appendix C.2): e.g. 500
episodes x 200 steps for fitting H0 probes and a separate 200-episode set for
imagination probes, collected once and shared by all models.
"""
import numpy as np

from . import make, state_labels


def collect_trajectories(env_name, num_episodes, env_seed=42, action_seed=0,
                         episode_length=200, progress=False, **env_kwargs):
    """Roll out a uniform random policy and record images, actions and labels.

    Args:
        env_name: registry name, e.g. ``"noisyshape"``.
        num_episodes: number of episodes.
        env_seed: seed of the environment RNG (positions, backgrounds, rewards).
        action_seed: seed of the random policy.
        episode_length: steps per episode (the paper uses 200).
        **env_kwargs: forwarded to :func:`shapeenv.make` (e.g. ``davis_path``).

    Returns:
        images:   uint8 ``(E, T+1, H, W, 3)``
        actions:  float32 one-hot ``(E, T, A)``; ``actions[:, t]`` leads from
                  frame ``t`` to frame ``t+1``
        is_first: float32 ``(E, T+1)``
        labels:   dict of ``(E, T+1)`` arrays, one per label key of the env
                  (same keys as :func:`shapeenv.state_labels`).
    """
    env = make(env_name, seed=env_seed, **env_kwargs)
    num_actions = env.action_space.n
    rng = np.random.RandomState(action_seed)
    iterator = range(num_episodes)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"Collecting {env_name}")
    all_images, all_actions, all_is_first, all_labels = [], [], [], []
    for _ in iterator:
        obs = env.reset()
        imgs, isf, acts = [obs['image']], [float(obs['is_first'])], []
        labs = [state_labels(env)]
        for _ in range(episode_length):
            ai = rng.randint(0, num_actions)
            a = np.zeros(num_actions, dtype=np.float32)
            a[ai] = 1.0
            obs, _, _, _, _ = env.step(ai)
            imgs.append(obs['image'])
            isf.append(float(obs.get('is_first', False)))
            acts.append(a)
            labs.append(state_labels(env))
        all_images.append(np.stack(imgs))
        all_actions.append(np.stack(acts))
        all_is_first.append(np.array(isf, dtype=np.float32))
        all_labels.append(labs)
    env.close()
    keys = all_labels[0][0].keys()
    labels = {k: np.array([[l[k] for l in ep] for ep in all_labels]) for k in keys}
    return (np.stack(all_images), np.stack(all_actions), np.stack(all_is_first), labels)
