"""Locating the DAVIS 2017 frames used as distractor backgrounds."""
import os


def resolve_davis_path(davis_path=None):
    """Return the DAVIS ``JPEGImages/480p`` directory to load backgrounds from.

    An explicit ``davis_path`` wins; otherwise the ``DAVIS_DIR`` environment
    variable is used. If neither points to an existing directory the
    environments fall back to uniform random-noise backgrounds and print a
    warning (this fallback was never used for any result in the paper).
    """
    if davis_path is None:
        davis_path = os.environ.get("DAVIS_DIR", "")
    return davis_path
