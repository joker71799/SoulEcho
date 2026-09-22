"""
疗愈 Agent 对话大模型（Chat LLM）配置。

- 本文件负责 Agent 生成疗愈回复时调用的对话大模型。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

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


def _get_float(key: str, default: float) -> float:
    """把字符串形式的浮点数转成 Python float，用于 temperature 这类数值配置。
    非法值直接抛异常，不静默回退到默认值。"""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(f"配置项 {key} 需要是数字，当前值为 {raw!r}。") from None


# 百炼 OpenAI 兼容端点与密钥。
# 默认值为通用 DashScope 端点；.env 里可覆盖成专属（maas）端点。
AGENT_LLM_BASE_URL = _get(
    "AGENT_LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
AGENT_LLM_API_KEY = _require("DASHSCOPE_API_KEY")

# 模块级单例：疗愈场景偏好稳定，进程内复用同一个客户端即可，无需每次请求重建。
chat_llm = ChatOpenAI(
    model=_get("AGENT_LLM_MODEL", "qwen-plus"),
    api_key=AGENT_LLM_API_KEY,
    base_url=AGENT_LLM_BASE_URL,
    temperature=_get_float("AGENT_LLM_TEMPERATURE", 0.8),
)