"""统一日志配置。

为什么单独抽一个模块：日志是生产排障的第一现场，
各模块只要 `logging.getLogger(__name__)` 即可，不用重复配置格式与级别。
"""
from __future__ import annotations

import logging
import os
import sys


def setup_logging() -> None:
    # 级别用环境变量控制：开发想安静就 LOG_LEVEL=WARNING，排障时 DEBUG
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    if root.handlers:  # 防止重复初始化导致日志打两遍
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)
