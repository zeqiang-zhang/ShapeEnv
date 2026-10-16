"""Render a few random-policy steps of every environment into one image.

    python examples/gallery.py --davis_path /path/to/DAVIS/JPEGImages/480p --out gallery.png
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

import shapeenv

ROWS = [
    ("shape", {}), ("shapefixed", {}), ("noisyshape", {}), ("noisyshapeseq", {}),
    ("noisyshapemotion", {}), ("noisytwoshape", {}), ("passiveshape", {}),
    ("texnoisyshape", {"level": 1}), ("texnoisyshape", {"level": 2}), ("texnoisyshape", {"level": 3}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis_path", default=os.environ.get("DAVIS_DIR"))
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--out", default="gallery.png")
    args = ap.parse_args()

    label_w, size = 150, 64 * args.scale
    canvas = Image.new("RGB", (label_w + args.steps * (size + 2), len(ROWS) * (size + 2)), "white")
    draw = ImageDraw.Draw(canvas)
    rng = np.random.RandomState(args.seed)
    for r, (name, kw) in enumerate(ROWS):
        if name in shapeenv.USES_DAVIS:
            kw = {**kw, "davis_path": args.davis_path}
        env = shapeenv.make(name, seed=args.seed, **kw)
        frames = [env.reset()["image"]]
        for _ in range(args.steps - 1):
            frames.append(env.step(rng.randint(env.action_space.n))[0]["image"])
        title = name + (f" L{kw['level']}" if "level" in kw else "")
        draw.text((6, r * (size + 2) + size // 2 - 6), title, fill="black")
        for c, f in enumerate(frames):
            img = Image.fromarray(f).resize((size, size), Image.NEAREST)
            canvas.paste(img, (label_w + c * (size + 2), r * (size + 2)))
    canvas.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
