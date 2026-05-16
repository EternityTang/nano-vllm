from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import torch


F = TypeVar("F", bound=Callable)


def maybe_compile(fn: F) -> F:
    try:
        return torch.compile(fn)  # type: ignore[return-value]
    except ImportError:
        return fn
