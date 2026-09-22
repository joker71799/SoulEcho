"""
全项目日志统一配置模块（基于 loguru）。

职责：
1. 在模块首次被导入时完成一次性的 loguru 初始化（控制台 + 滚动文件双通道）。
2. 对外暴露统一的 `logger`，各业务模块只 import 它，不直接碰 loguru 全局配置。
3. 提供 `mask_secret` 工具，日志中输出 API Key 等敏感信息时强制脱敏。

设计约定：
- 日志级别可用 .env / 系统环境变量 LOG_LEVEL 覆盖，默认 INFO。
- 文件日志落在项目根 logs/ 下，按大小滚动并保留一段时间，目录已加入 .gitignore。
- 采用导入即初始化：保证任何模块拿到 logger 时配置已就绪，不依赖调用顺序。
"""

import os
import sys
from pathlib import Path

from loguru import logger as _logger

# 本文件位于 backend/app/config/，向上 3 级即项目根目录 SoulEcho/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = PROJECT_ROOT / "logs"


def mask_secret(value: str) -> str:
    """敏感信息脱敏：保留头 4 位和尾 2 位，中间打码，用于日志输出 API Key 等。"""
    if not value or len(value) <= 8:
        return "****"
    return f"{value[:4]}****{value[-2:]}"


def preview(text: str, limit: int = 50) -> str:
    """长文本截断预览：超过 limit 个字符则截断并补省略号。

    仅用于日志中输出 AI 回复等长内容；用户 query 及本身简短的字段不做截断。
    """
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + "…"


def _setup_logging() -> None:
    """初始化 loguru：移除默认 handler，注册控制台与滚动文件两个输出通道。"""
    level = os.getenv("LOG_LEVEL", "INFO").upper()

    # 格式化：时间 | 级别 | 应用标签 | 模块:函数:行号 | 消息，异常自动附带堆栈
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<magenta>[{extra[app]}]</magenta> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    _logger.remove()  # 去掉 loguru 默认的 stderr handler，统一走下面的配置
    # 通道 1：控制台，开发调试用，带颜色
    _logger.add(sys.stderr, level=level, format=fmt, colorize=True)
    # 通道 2：滚动文件，排查历史问题用；enqueue=True 保证多进程（uvicorn reload）下写日志不冲突
    _logger.add(
        LOG_DIR / "soulecho_{time:YYYY-MM-DD}.log",
        level=level,
        format=fmt,
        rotation="10 MB",   # 单文件超过 10MB 自动切分
        retention="14 days",  # 日志保留 14 天
        encoding="utf-8",
        enqueue=True,
        backtrace=True,     # 异常记录完整调用栈
        diagnose=True,      # 堆栈中附带变量值（本地调试利器；上生产建议关掉防泄漏）
    )


# 导入即完成初始化（幂等：重复 import 不会重复注册 handler）
_setup_logging()

# 对外暴露的统一 logger；bind 出 app 标签，便于和 uvicorn 等第三方日志区分
logger = _logger.bind(app="soulecho")
