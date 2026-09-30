"""Allow ``python -m odockmcode`` to run the command-line interface."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
