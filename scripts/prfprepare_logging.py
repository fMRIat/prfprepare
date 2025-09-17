# prfprepare_logging.py

import os
import sys
from dataclasses import dataclass
from typing import Optional


def _basename(file_path: str) -> str:
    """
    Extract basename from file path, ensuring .py extension.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    str
        Basename with .py extension.
    """
    base = os.path.basename(file_path)
    return base if base.endswith(".py") else f"{base}.py"


@dataclass
class _Logger:
    file: str
    verbose: bool = False

    def set_verbose(self, verbose: bool) -> None:
        """
        Set the verbosity level for debug messages.

        Parameters
        ----------
        verbose : bool
            Whether to enable verbose debug output.
        """
        self.verbose = bool(verbose)

    # Always-visible important messages
    def info(self, msg: str, *args) -> None:
        """
        Log an informational message (always visible).

        Parameters
        ----------
        msg : str
            Message format string.
        *args
            Arguments for string formatting.
        """
        self._emit("", msg, *args, stream=sys.stdout)

    # Verbose-only messages
    def debug(self, msg: str, *args) -> None:
        """
        Log a debug message (only visible when verbose=True).

        Parameters
        ----------
        msg : str
            Message format string.
        *args
            Arguments for string formatting.
        """
        if self.verbose:
            self._emit("", msg, *args, stream=sys.stdout)

    # Warnings (always visible, stderr)
    def warn(self, msg: str, *args) -> None:
        """
        Log a warning message (always visible, to stderr).

        Parameters
        ----------
        msg : str
            Message format string.
        *args
            Arguments for string formatting.
        """
        self._emit("WARNING: ", msg, *args, stream=sys.stderr)

    # Errors (always visible, stderr)
    def error(self, msg: str, *args) -> None:
        """
        Log an error message (always visible, to stderr).

        Parameters
        ----------
        msg : str
            Message format string.
        *args
            Arguments for string formatting.
        """
        self._emit("ERROR: ", msg, *args, stream=sys.stderr)

    def _emit(self, level_prefix: str, msg: str, *args, stream) -> None:
        """
        Emit a formatted log message to the specified stream.

        Parameters
        ----------
        level_prefix : str
            Prefix indicating log level (e.g., "WARNING: ").
        msg : str
            Message format string.
        *args
            Arguments for string formatting.
        stream : file-like
            Output stream (stdout or stderr).
        """
        prefix = f"[prfprepare/{_basename(self.file)}] "
        try:
            formatted = msg % args if args else msg
        except Exception:
            # Fallback if formatting fails
            formatted = f"{msg} {' '.join(map(str, args))}".strip()
        print(prefix + level_prefix + formatted, file=stream, flush=True)


def get_logger(file: str, verbose: Optional[bool] = None) -> _Logger:
    """
    Create and return a logger instance for the specified file.

    Parameters
    ----------
    file : str
        File path or name for logger identification.
    verbose : bool or None, optional
        Whether to enable verbose output. If None, checks PRFPREPARE_VERBOSE
        environment variable.

    Returns
    -------
    _Logger
        Configured logger instance.
    """
    if verbose is None:
        # env var takes precedence when explicit flag isn't provided
        env_v = os.getenv("PRFPREPARE_VERBOSE", "0").strip().lower()
        verbose = env_v in ("1", "true", "yes", "on")
    return _Logger(file=file, verbose=bool(verbose))
