"""General logger."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def hp2b_logger(log_level: str = "INFO") -> logging.Logger:
    # FORMAT = '\n%(asctime)s - %(name)s - %(levelname)s\n\t%(message)s\n'
    FORMAT = "%(message)s"

    logging.basicConfig(
        level=log_level,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler()],
    )

    return logging.getLogger("halfpipe2bids")
