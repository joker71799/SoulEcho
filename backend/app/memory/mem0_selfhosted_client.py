"""自托管 Mem0 REST 服务器的「最小可闭环」客户端。

背景
----
官方自带的 ``memory.MemoryClient`` 指向托管平台 https://api.mem0.ai ，说的是平台版
的 ``/v1/`` + ``/v3/`` 接口；而 ``server/`` 里的自托管服务器路由没有版本前缀
（``POST /memories``、``POST /search`` ……），也根本没有 ``/v1/ping/``。两者路径
和鉴权都对不上，所以官方 SDK 无法直接对接自托管服务器。

本文件就是对准自托管服务器 ``server/main.py`` 的路由，手写的一个极轻量客户端，
只覆盖最核心的一条闭环链路：

    add（写入）  →  search（检索）  →  get_all（列出）  →  delete（删除）

设计取舍（刻意保持“最小”）
--------------------------
* 只支持 ``X-API-Key`` 鉴权（按用户 key 或兼容的 ADMIN_API_KEY），不含 JWT 登录流程。
* 只依赖 httpx（memory 已有的依赖），不引入额外第三方库。
* 不做分页封装、不做强类型模型，返回值直接就是服务器原始的 JSON dict。
* 出错时统一抛出 ``Mem0APIError``，带上 HTTP 状态码和服务器返回的 detail。

快速上手
--------
    from mem0_selfhosted_client import Mem0ServerClient

    # 先按 server/README.md 启动自托管服务器，并拿到一个 API key
    c = Mem0ServerClient("http://localhost:8888", api_key="m0sk_你的key")

    # 1) 写入一条记忆（传字符串会自动包装成 user 消息）
    c.add("我最喜欢吃新鲜的蔬菜披萨。", user_id="alice")

    # 2) 语义检索
    hits = c.search("蔬菜", user_id="alice")
    print(hits)

    # 3) 列出该用户的全部记忆
    print(c.get_all(user_id="alice"))

    # 4) 删除单条记忆（memory_id 从上面结果里取）
    # c.delete("<memory_id>")

依赖
----
    pip install httpx      # 如果你已通过 memory 环境使用，通常已装好
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import httpx

# 一条消息的字典结构，例如 {"role": "user", "content": "......"}
# 类型别名
Message = Union[str, Dict[str, str], List[Dict[str, str]]]


class Mem0APIError(RuntimeError):
    """当自托管服务器返回非 2xx 响应时抛出。

    属性：
        status_code: HTTP 状态码（例如 401、404、400）。
        detail:      服务器返回的错误详情（通常是 JSON 里的 detail 字段，
                     解析失败时退回原始响应文本）。
    """

    def __init__(self, status_code: int, detail: Any):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class Mem0ServerClient:
    """面向自托管 Mem0 服务器的最小客户端（同步）。

    实例化时会创建一个复用的 httpx.Client，把 base_url、鉴权头、超时统一配置好，
    后续所有请求都走它。用完建议用 ``with`` 语句或显式调用 close() 释放连接。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8888",
        api_key: Optional[str] = None,
        timeout: float = 300.0,
    ):
        """初始化客户端。

        参数：
            base_url: 自托管服务器的根地址。Docker Compose 默认把内部 8000 映射为
                      宿主机的 8888，所以本地一般是 http://localhost:8888。
                      末尾有没有斜杠都行，内部会做规范化。
            api_key:  X-API-Key 的值。可以是控制台创建的按用户 key（``m0sk_...``），
                      也可以是兼容的旧版 ADMIN_API_KEY。
                      提示：如果服务器是用 ``AUTH_DISABLED=true`` 启动的（仅本地开发），
                      可以不传 api_key；否则受保护接口会返回 401。
            timeout:  单次请求的超时秒数。写入/检索会调用 LLM，可能较慢，默认给了 300s。
        """
        # rstrip 去掉末尾斜杠，避免拼接出 http://host//memories 这种双斜杠路径
        self.base_url = base_url.rstrip("/")

        # 组装鉴权头：有 key 就带 X-API-Key，没有就不带（配合 AUTH_DISABLED 场景）
        headers: Dict[str, str] = {}
        if api_key:
            headers["X-API-Key"] = api_key

        # 复用同一个 httpx.Client：base_url 让后续只需写相对路径
        self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    # -------------------------------------------------------------- 记忆写入
    def add(
        self,
        messages: Message,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """POST /memories —— 从对话消息中提取并写入记忆。

        参数：
            messages: 支持三种写法，内部会统一规整成 list[dict]：
                      * 字符串     -> 自动包装成一条 user 消息
                      * 单个 dict  -> 包装成 [dict]
                      * list[dict] -> 原样使用（每个元素含 role/content）
            user_id / agent_id / run_id:
                      记忆的归属标识，三者至少传一个，否则服务器返回 400。
            metadata: 附带的自定义元数据（会存进向量库的 payload 里）。

        返回：
            服务器原始 JSON，通常形如 {"results": [{"id": ..., "event": "ADD", ...}]}。
        """
        payload = self._build_message_payload(
            messages,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata,
        )
        return self._request("POST", "/memories", json=payload)

    # -------------------------------------------------------------- 语义检索
    def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """POST /search —— 按语义检索某个范围内的记忆。

        参数：
            query:   检索文本（必填）。
            user_id / agent_id / run_id:
                     检索范围，会被合并进服务器要求的 ``filters`` 字段。
            top_k:   返回条数上限（可选）。

        返回：
            {"results": [{"id", "memory", "score", ...}]} 形式的字典。
        """
        # 把便捷实体参数合并进 filters，保持与新版服务器接口一致
        filters: Dict[str, Any] = {
            key: value
            for key, value in {
                "user_id": user_id,
                "agent_id": agent_id,
                "run_id": run_id,
            }.items()
            if value is not None
        }

        payload: Dict[str, Any] = {"query": query, "filters": filters}
        if top_k is not None:
            payload["top_k"] = top_k
        return self._request("POST", "/search", json=payload)

    # -------------------------------------------------------------- 列出记忆
    def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """GET /memories —— 按归属标识列出记忆。

        参数：
            user_id / agent_id / run_id: 三者至少传一个来限定范围，作为 query 参数发送。
            top_k: 返回条数上限（可选）。

        返回：
            {"results": [{"id", "memory", "user_id", ...}]} 形式的字典。
        """
        params = {
            key: value
            for key, value in {
                "user_id": user_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "top_k": top_k,
            }.items()
            if value is not None
        }
        return self._request("GET", "/memories", params=params)

    # -------------------------------------------------------------- 删除记忆
    def delete(self, memory_id: str) -> Dict[str, Any]:
        """DELETE /memories/{memory_id} —— 删除单条记忆。

        参数：
            memory_id: 记忆 ID（可从 add/search/get_all 的结果里拿到）。

        返回：
            {"message": "Memory deleted successfully"} 形式的字典。
        """
        return self._request("DELETE", f"/memories/{memory_id}")

    # ------------------------------------------------------------ 资源释放
    def close(self) -> None:
        """关闭底层 HTTP 连接。"""
        self._http.close()

    def __enter__(self) -> "Mem0ServerClient":
        # 支持 with 语法，进入时直接返回自身
        return self

    def __exit__(self, *exc: Any) -> None:
        # 退出 with 块时自动关闭连接
        self.close()

    # ------------------------------------------------------------ 内部工具
    @staticmethod
    def _build_message_payload(
        messages: Message,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """把用户传入的 messages 规整成服务器要求的结构，并附带非空参数。

        服务器端 MemoryCreate 期望：
            {"messages": [{"role", "content"}], "user_id"?, "agent_id"?, ...}
        """
        if isinstance(messages, str):
            # 字符串按“用户说的一句话”处理
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            # 单条消息字典 -> 包成列表
            messages = [messages]
        elif not isinstance(messages, list):
            # 其它类型一律拒绝，尽早报错
            raise ValueError("messages 必须是 str、dict 或 list[dict]")

        payload: Dict[str, Any] = {"messages": messages}
        # 只把非 None 的可选参数放进 payload，避免覆盖服务器默认值
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        return payload

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """统一发起请求 + 解析响应 + 抛出错误的入口。

        参数：
            method: HTTP 方法，如 "GET" / "POST" / "DELETE"。
            path:   相对 base_url 的路径，如 "/memories"。
            kwargs: 透传给 httpx 的 json / params 等参数。
        """
        resp = self._http.request(method, path, **kwargs)
        self._raise_for_status(resp)

        # 204 或空响应体没有 JSON，直接返回空字典，避免 json() 解析报错
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        """非 2xx 时转成 Mem0APIError；尽量从 JSON 里取 detail 字段。"""
        if resp.is_success:
            return
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            # 响应不是合法 JSON（比如网关 502 返回的 HTML），退回原始文本
            detail = resp.text
        raise Mem0APIError(resp.status_code, detail)