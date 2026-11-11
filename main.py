"""Entry point used by the GitHub Actions workflows: `python main.py [--mode ...]`."""

import sys

from bot.runner import main

if __name__ == "__main__":
    sys.exit(main())
