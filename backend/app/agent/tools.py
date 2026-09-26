"""把 Mem0 长期记忆检索封装成 ReAct 可供 LLM 自主调用的工具。

设计意图：
- 原 retrieve_memory 节点每轮无条件检索，简单情绪陪伴也照查，白耗一次向量检索与延迟。
- 改成工具后，"这轮到底要不要翻历史"交给模型自主推理（ReAct 的行动步），
  只在确有需要用户过去背景来共情时才发起检索，省下无谓往返。
"""
import time
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from config.mem0_config import mem0_client, MEM0_SEARCH_MIN_SCORE
from config.logging_config import logger, preview

# 命中时加个帽：把检索结果框成封闭世界（只有这些是事实），拦住它顺势“补点细节”的冲动。
MEMORY_HEADER = "【检索到的历史记忆，关于过去你只知道以下这些】\n"

# 未命中时回给模型的观察结果：既是事实陈述，也顺带重申不得编造。
NO_MEMORY_FOUND = (
    "（没有找到相关历史记忆：该用户从未提过这件事。"
    "请坦诚承认不记得并邀请对方告诉你，严禁编造任何具体细节）"
)


@tool
def search_memory(
    query: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """检索该用户跨会话的长期记忆（历史经历、情感画像）。

    用户直接问你“记不记得”某事、或询问自己的偏好/经历时，必须调用本工具核实，
    不得凭印象回答；若本轮只是简单情绪陪伴、不依赖历史信息，才可以不调用。
    query 必须语义完整、不含“他/那件事/又”等指代词。

    返回内容就是你能知道的关于过去的全部事实：
    若返回“没有找到相关历史记忆”，则该用户真的没提过，不得自行编造。
    """
    user_id = state["user_id"]
    conversation_id = state.get("conversation_id", "")
    started = time.perf_counter()
    previous_memories = mem0_client.search(query, user_id=user_id)

    # 向量检索不看相关度、恒回 top_k 条：库里没相关记忆时会把语义最近邻的无关内容
    # 塞回来（问“喝什么”返回“喜欢桂花”），模型据此就会“记得”一杯不存在的茶。
    # 官方 API 保证每条记忆都带 score，故在交给模型前先按分数过一道，不达标的当“没查到”。
    results = previous_memories.get("results") or []
    pairs = [(float(m["score"]), m["memory"]) for m in results if m.get("memory")]
    relevant = sorted(
        (pair for pair in pairs if pair[0] >= MEM0_SEARCH_MIN_SCORE),
        key=lambda pair: pair[0],
        reverse=True,
    )
    max_score = max((score for score, _ in pairs), default=0.0)

    memory_context = (
        MEMORY_HEADER + "\n".join(text for _, text in relevant) if relevant else ""
    )

    # 命中记忆逐条截断预览（避免长文本撞日志），带分数方便调优阈值
    hit_detail = (
        " | ".join(f"{score:.3f}:{preview(text)}" for score, text in relevant)
        if relevant
        else "无"
    )
    logger.info(
        "ReAct 记忆检索完成 user_id={} conversation_id={} hit={}/{} max_score={:.3f} min_score={} "
        "elapsed={:.0f}ms query={!r} memories={}",
        user_id, conversation_id, len(relevant), len(pairs), max_score, MEM0_SEARCH_MIN_SCORE,
        (time.perf_counter() - started) * 1000, preview(query), hit_detail,
    )
    # 供模型“观察”的工具返回值；无命中给明确信号，避免模型编造历史
    observation = memory_context or NO_MEMORY_FOUND

    # Step 2：命中时把过阈值的记忆条目写回 state.retrieved_memories（纯生产端，本轮无消费方）；
    # 无命中不写，保持 crisis_router 每轮重置后的 []。工具需同时回观察、又写状态，
    # 故返回 Command：观察包在 ToolMessage 里回给 ReAct，记忆条目随 update 合并进图状态。
    update: dict = {"messages": [ToolMessage(observation, tool_call_id=tool_call_id)]}
    if relevant:
        update["retrieved_memories"] = [text for _, text in relevant]
    return Command(update=update)


# 供 CF / 闲聊节点 bind_tools、ToolNode 注册的唯一数据源
HEALING_TOOLS = [search_memory]
