"""
logging_setup.py
================

Single place to configure **both** the std-lib `logging` package and
Loguru so that:

• every module can just do `import logging; logger = logging.getLogger(__name__)`
  and get consistent formatting;
• Loguru’s rich formatting / rotation is still available for sinks
  that need it (e.g. file logs);
• configuration happens **once** (idempotent).

Call `configure_logging()` as early as possible (it is invoked in
`webhook_server.py` and the trainer entry-point).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Literal

from loguru import logger

# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
# Dedicated sink for tickets which were not auto-merged.  Environment variable
# can override the default location used by tests and Docker.
unmerged_path = os.getenv("UNMERGED_LOG", "/data/unmerged_tickets.log")

# --------------------------------------------------------------------------- #
#  Utility
# --------------------------------------------------------------------------- #
_LOG_INITIALISED = False

_unmerged_sink_added = False

if not _unmerged_sink_added:
    logger.add(unmerged_path, level="INFO", enqueue=True,
               format="{time} {message}")
    _unmerged_sink_added = True

def _intercept_stdlib(level: int | None = None) -> None:
    """
    Redirect std-lib logging calls into Loguru so all logs
    benefit from the same formatting and rotation.
    """
    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                lvl: int | str = logger.level(record.levelname).name
            except ValueError:
                lvl = record.levelno
            # Find caller from where originated the log record
            frame, depth = logging.currentframe(), 2
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(lvl, record.getMessage())

    logging.root.handlers = [InterceptHandler()]
    if level is not None:
        logging.root.setLevel(level)


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def configure_logging(
    *,
    console_level: str | int = "INFO",
    file_level: str | int = "INFO",
    file_path: str | Path | None = None,
    rotation: str | int | None = "10 MB",
    retention: str | int | None = "7 days",
    enqueue: bool = True,
    diagnose: bool = False,
    colorize: Literal[True, False] = True,
) -> None:
    """
    Initialise Loguru & stdlib logging once. Subsequent calls
    are no-ops.

    Parameters
    ----------
    console_level
        Minimum level for stderr sink.
    file_level
        Minimum level for the file sink (if *file_path* is given).
    file_path
        Optional file path.  If omitted, only stderr is used.
    rotation
        Loguru rotation config (bytes str like "10 MB" or "00:00").
    retention
        Retention policy for rotated files.
    enqueue
        Whether to enqueue logs (required in multi-process).
    diagnose
        Forward Loguru's internal diagnositics (noisy).
    colorize
        ANSI colourise console sink.
    """
    global _LOG_INITIALISED
    if _LOG_INITIALISED:
        return
    _LOG_INITIALISED = True

    # 1) hijack std-lib → loguru
    _intercept_stdlib()

    # 2) console sink
    logger.remove()
    logger.add(
        sys.stderr,
        level=console_level,
        colorize=colorize,
        backtrace=False,
        diagnose=diagnose,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        enqueue=enqueue,
    )

    # 3) optional file sink
    if file_path:
        log_path = Path(file_path).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path,
            level=file_level,
            rotation=rotation,
            retention=retention,
            enqueue=enqueue,
            backtrace=False,
            diagnose=diagnose,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{line} - {message}",
        )

    # expose to std-lib modules
    logging.getLogger(__name__).debug("Logging initialised")
