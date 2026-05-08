from __future__ import annotations

"""Runtime warning/log suppression helpers for local LeRobot workflows.

This module is intentionally conservative:
- keep ERROR / CRITICAL / exceptions visible
- suppress only known repetitive fallback / deprecation / experimental warnings
- allow instant opt-out with ``LEROBOT_DEBUG_WARNINGS=1``
"""

import json
import logging
import os
import re
import warnings
from typing import Iterable

_DEBUG_ENV_VARS = ("LEROBOT_DEBUG_WARNINGS", "LEROBOT_DEBUG")

# NOTE:
# Keep these patterns narrow. The goal is not to silence all warnings, only
# noisy repeats that do not affect the actual runtime path when fallback logic
# is already working.
_SUPPRESSED_LOG_PATTERNS = [
    r".*'torchcodec' is not available in your platform, falling back to 'pyav'.*",
    r".*torchvision import failed; video decoding will use PyAV fallback paths:.*",
    r".*torchvision\.io\.VideoReader is unavailable in this build; falling back to direct PyAV decoding.*",
]


class SuppressKnownWarningsFilter(logging.Filter):
    """Suppress known repetitive WARNING/INFO log messages only.

    ERROR/CRITICAL are always kept.
    """

    def __init__(self, patterns: Iterable[str]):
        super().__init__()
        self._patterns = [re.compile(p) for p in patterns]

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True

        try:
            message = record.getMessage()
        except Exception:
            return True

        return not any(pattern.match(message) for pattern in self._patterns)


def is_debug_warning_mode() -> bool:
    for env_name in _DEBUG_ENV_VARS:
        value = os.getenv(env_name, "").strip().lower()
        if value in {"1", "true", "yes", "on", "debug"}:
            return True
    return False


def configure_runtime_warnings() -> None:
    """Apply conservative warning suppression for common local workflows.

    Safe by design:
    - import failures, camera failures, dataset write failures, model load
      failures, and tracebacks are not hidden
    - only targeted warning message patterns are suppressed
    - DEBUG mode restores normal warning visibility
    """

    if is_debug_warning_mode():
        warnings.simplefilter("default")
        return

    warnings.simplefilter("default")

    # Python warnings: narrow message-based suppression only.
    warnings.filterwarnings(
        "ignore",
        message=r".*The video decoding and encoding capabilities of torchvision.*deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*video decoding.*deprecated.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*video decoding.*deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*'register_feature' is experimental and might be subject to breaking changes.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*pkg_resources is deprecated as an API.*",
        category=DeprecationWarning,
    )

    root_logger = logging.getLogger()
    message_filter = SuppressKnownWarningsFilter(_SUPPRESSED_LOG_PATTERNS)
    root_logger.addFilter(message_filter)
    for handler in root_logger.handlers:
        handler.addFilter(message_filter)


def log_structured_summary(title: str, payload: dict) -> None:
    """Emit concise structured summary through logging instead of noisy prints."""

    logging.info("%s\n%s", title, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
