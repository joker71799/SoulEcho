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


# ------------------------------------------------------------
# 3. 构建 Mem0 自托管服务客户端（对外暴露使用）
# ------------------------------------------------------------
MEM0_SERVER_BASE_URL = _require("MEM0_SERVER_BASE_URL")
MEM0_SERVER_API_KEY = _require("MEM0_SERVER_API_KEY")

# 模块级单例：复用同一个 httpx 连接池，进程内无需每次请求重建
mem0_client = Mem0ServerClient(
    base_url=MEM0_SERVER_BASE_URL,
    api_key=MEM0_SERVER_API_KEY,
)