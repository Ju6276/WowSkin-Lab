"""Compatibility entry point; see python -m wowskin_lab --help."""
from wowskin_lab.legacy.cli import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
