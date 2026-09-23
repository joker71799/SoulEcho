"""
TypeSafe Jev（System One 决策模型）配置加载模块。

职责（本文件是 Jev 配置的唯一加载入口）：
1. 从项目根 .env 读取调用 Jev 所需的 API 密钥与模型标识。
2. 构建并暴露一个可复用的 TypeSafeClassifier 单例，供危机识别等分类场景直接调用。

说明：
- 与 llm_config.py / mem0_config.py 一致：端点与密钥集中在 .env，代码中不硬编码。
- Jev 不生成文本，只返回带概率与置信度的结构化判断，因此只用于「识别/分类」，
  不用于生成疗愈回复。
- classifier 实例内部持有 httpx 连接池，进程内长期复用即可，勿每次请求重建。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from config.logging_config import logger, mask_secret
from langchain_typesafe import TypeSafeClassifier

# 本文件位于 backend/app/config/，向上 3 级即项目根目录 SoulEcho/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"

# override=False：系统环境变量已存在同名值时优先用系统的，方便临时覆盖
load_dotenv(ENV_PATH, override=False)


def _require(key: str) -> str:
    """读取必填环境变量，缺失时直接报错，避免带着 None 往下走难以排查。"""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"缺少必填配置项 {key}，请在 {ENV_PATH} 中设置后重试。")
    return value


def _get(key: str, default: str) -> str:
    """读取可选环境变量，缺失时用默认值。"""
    return os.getenv(key, default)


# 官方模型别名 jev-latest 指向当前稳定版；如需锁定版本可在 .env 显式指定 JEV_MODEL。
JEV_MODEL = _get("JEV_MODEL", "jev-latest")
TYPESAFE_API_KEY = _require("TYPESAFE_API_KEY")

# 端点集中在 .env 管理：默认走官方 https://api.typesafe.ai，
# 配置为 https://openrouter.ai/api 时，langchain-typesafe 会拼出
# https://openrouter.ai/api/v1/systemone 走 OpenRouter 兼容端点。
TYPESAFE_BASE_URL = _get("TYPESAFE_BASE_URL", "https://api.typesafe.ai")

# 模块级单例：危机识别每轮都要调用，复用同一连接池避免重复握手
jev_classifier = TypeSafeClassifier(
    model=JEV_MODEL, api_key=TYPESAFE_API_KEY, base_url=TYPESAFE_BASE_URL
)

# 启动时记录一次生效配置摘要（密钥脱敏），方便确认 Jev 配置是否按预期加载
logger.info(
    "TypeSafe Jev 配置加载完成 model={} base_url={} api_key={}",
    JEV_MODEL, TYPESAFE_BASE_URL, mask_secret(TYPESAFE_API_KEY),
)
