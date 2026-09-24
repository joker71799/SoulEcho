"""
Mem0（自托管 HTTP 服务）配置加载模块。

职责（本文件是 Mem0 配置的唯一加载入口）：
1. 从项目根 .env 读取连接自托管 Mem0 服务器所需的地址与 API 密钥。
2. 构建并暴露一个可复用的 Mem0ServerClient 单例，供外部直接调用。

说明：
- 已彻底放弃 OSS 函数库（进程内 Memory.from_config）模式，改为通过 HTTP 客户端
  调用独立部署的自托管 Mem0 服务，实现服务解耦。
- 与 llm_config.py 一致：所有端点/密钥集中在 .env，代码中不硬编码。
- 具体路径、鉴权等细节由 memory.mem0_selfhosted_client.Mem0ServerClient 负责，
  本模块只提供它所需的 base_url 与 api_key。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from memory.mem0_selfhosted_client import Mem0ServerClient
from config.logging_config import logger, mask_secret

# ------------------------------------------------------------
# 1. 定位并加载 .env
# ------------------------------------------------------------
# 本文件位于 backend/app/config/，向上 3 级即项目根目录 SoulEcho/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"

# override=False：若系统环境变量已存在同名值，优先用系统的，方便临时覆盖
load_dotenv(ENV_PATH, override=False)


# ------------------------------------------------------------
# 2. 环境变量读取
# ------------------------------------------------------------
def _require(key: str) -> str:
    """读取必填环境变量，缺失时直接报错，避免带着 None 往下走难以排查。"""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"缺少必填配置项 {key}，请在 {ENV_PATH} 中设置后重试。")
    return value


def _get_float_in_0_1(key: str, default: float) -> float:
    """读取 [0,1] 区间的浮点配置（相关度分数一类的阈值）。

    缺失或空值用默认；填了非法值（非数字或越界）直接抛异常，不静默回退。"""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        raise RuntimeError(f"配置项 {key} 需要是数字，当前值为 {raw!r}。") from None
    if not 0.0 <= value <= 1.0:
        raise RuntimeError(f"配置项 {key} 需要在 0~1 之间，当前值为 {raw!r}。")
    return value


# ------------------------------------------------------------
# 3. 构建 Mem0 自托管服务客户端（对外暴露使用）
# ------------------------------------------------------------
MEM0_SERVER_BASE_URL = _require("MEM0_SERVER_BASE_URL")
MEM0_SERVER_API_KEY = _require("MEM0_SERVER_API_KEY")

# 语义检索的相关度下限。向量检索是「无论如何都返回 top_k 条」，库里没有对应记忆时
# 也会塞来语义最近邻的无关内容（问「喝什么」返回「喜欢桂花」），模型据此就会编造
# 「我记得你喝桂花乌龙」。低于此分数的记忆一律视为无关丢弃，让工具能给出真实的
# 「没找到」信号。取值依赖具体向量库的打分口径，可结合检索日志中的 max_score 调优。
MEM0_SEARCH_MIN_SCORE = _get_float_in_0_1("MEM0_SEARCH_MIN_SCORE", 0.5)

# 模块级单例：复用同一个 httpx 连接池，进程内无需每次请求重建
mem0_client = Mem0ServerClient(
    base_url=MEM0_SERVER_BASE_URL,
    api_key=MEM0_SERVER_API_KEY,
)

# 启动时记录一次生效配置摘要（密钥脱敏），方便确认 Mem0 服务指向是否正确
logger.info(
    "Mem0 自托管服务配置加载完成 base_url={} min_score={} api_key={}",
    MEM0_SERVER_BASE_URL, MEM0_SEARCH_MIN_SCORE, mask_secret(MEM0_SERVER_API_KEY),
)