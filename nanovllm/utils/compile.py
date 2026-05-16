from __future__ import annotations

# 中文说明：
# P-1 阶段引入的 torch.compile 兼容包装器，用于让核心 layer 在支持编译时保持原有加速路径，在环境不支持或导入失败时回退到普通 Python 函数。
# 该工具的作用是降低 smoke test 和能力探针的环境敏感度；它不改变算子语义，也不引入新的运行时依赖。

from collections.abc import Callable
from typing import TypeVar

import torch


F = TypeVar("F", bound=Callable)


def maybe_compile(fn: F) -> F:
    try:
        return torch.compile(fn)  # type: ignore[return-value]
    except ImportError:
        return fn
