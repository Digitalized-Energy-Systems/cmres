"""Centralized logging configuration for cmres.

All cmres modules obtain their logger via::

    import logging
    log = logging.getLogger(__name__)

Call ``cmres.log.setup()`` once from the entry-point script to attach
handlers.  Until then every log record is silently swallowed by the default
``logging.lastResort`` handler (WARNING+ to stderr only).

Console output (INFO by default)
---------------------------------
::

    10:23:45  INFO     cmres.resilience.mc           [MC] n=  200  ess=100.0 ...
    10:25:45  INFO     cmres.resilience.mc           [MC] Converged after 412 runs
    10:25:45  INFO     cmres.run                     Finished in 2.1 min

File output (DEBUG, written to <out_dir>/run.log)
--------------------------------------------------
Contains everything the console shows plus solver-level chatter that would
be too noisy for a terminal (one line per energy-flow solve, per MC run).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

# ── Formatters ────────────────────────────────────────────────────────────────

_FMT_CONSOLE = "%(asctime)s  %(levelname)-7s  %(name)-32s  %(message)s"
_FMT_FILE = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"
_DATEFMT_CON = "%H:%M:%S"
_DATEFMT_FIL = "%Y-%m-%d %H:%M:%S"

# ── Public API ────────────────────────────────────────────────────────────────


def setup(
    console_level: int = logging.INFO,
    log_file: Optional[Path] = None,
    file_level: int = logging.DEBUG,
) -> None:
    """Attach handlers to the ``cmres`` root logger.

    Safe to call multiple times — subsequent calls are no-ops if handlers are
    already attached.

    Parameters
    ----------
    console_level:
        Minimum level written to stdout (default INFO).
    log_file:
        Path to write the full debug log.  ``None`` → no file output.
    file_level:
        Minimum level written to the log file (default DEBUG).
    """
    root = logging.getLogger("cmres")
    root.setLevel(logging.DEBUG)  # handlers do the per-destination filtering

    if root.handlers:
        return  # already set up — idempotent

    # ── Console ───────────────────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter(_FMT_CONSOLE, datefmt=_DATEFMT_CON))
    root.addHandler(ch)

    # ── File (optional) ───────────────────────────────────────────────────────
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(logging.Formatter(_FMT_FILE, datefmt=_DATEFMT_FIL))
        root.addHandler(fh)

    root.debug(
        "cmres logging initialised (console=%s, file=%s)",
        logging.getLevelName(console_level),
        str(log_file) if log_file else "none",
    )
