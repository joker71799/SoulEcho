"""回复收尾节点：全图唯一的 reply 写者与唯一的记忆沉淀者（合并原危机生成 + ReAct 收尾）。

危机与正常两条链路都汇聚到这里收口，按分支确定 reply 来源后共用同一条记忆写入：
- 危机链路（is_crisis=True）：crisis_router 命中后直达，reply 用受控固定安全话术 + 热线卡，
  不走自由生成（危机场景不容高温度即兴/幻觉），但仍把本轮问答写入 Mem0 保留危机信号；
- 正常链路（ReAct 收敛后）：检索与生成都已在 ReAct 循环完成，这里只从 messages 里
  取最后一条有正文、非工具调用的 AI 回复作为最终 reply。

两分支职责一致——“定本轮 reply + 一定沉淀记忆”，故 reply 由本节点单点写入，
守住“单写者”约定，供 main.py 直接取用。
"""
import time

from langchain_core.messages import AIMessage, HumanMessage

from agent.crisis.hotlines import build_crisis_card
from agent.crisis.templates import CRISIS_TEMPLATE
from agent.graph.state import AgentState
from config.logging_config import logger, preview

# ==========================================
# 长期记忆组件 (Mem0 - 自托管 HTTP 服务)
# ==========================================
from config.mem0_config import mem0_client


def finalize_reply_node(state: AgentState):
    """本轮收尾：按分支确定 reply（危机话术 / ReAct 产物）并一定沉淀本轮记忆。"""
    # 本轮用户倾诉：反向找最后一条 HumanMessage（危机轮即当前倾诉，正常轮穿过工具消息）
    user_input = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
    )

    # 危机分支：固定安全模板 + 热线卡，刻意不调疗愈 LLM
    if state.get("is_crisis"):
        level = state.get("crisis_level", "risk")
        reply = CRISIS_TEMPLATE.get(level) or CRISIS_TEMPLATE["risk"]
        logger.warning(
            "命中危机干预分支，返回安全模板 user_id={} level={}",
            state.get("user_id", "default_user"), level,
        )
        # 危机话术是新造的模板、不在 messages 里，需追加为 AI 消息以保持多轮历史一致
        _persist_turn_memory(state, user_input, reply)
        # crisis_card 的清空已由首节点 crisis_router 统一负责，本分支只负责命中时写入
        return {
            "reply": reply,
            "crisis_card": build_crisis_card(level=level),
            "messages": [AIMessage(content=reply)],
        }

    # 正常分支：ReAct 已生成回复，反向取最后一条“有正文且非工具调用”的 AIMessage；
    reply = next(
        (
            m.content for m in reversed(state["messages"])
            if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None)
        ),
        "",
    )
    # reply 已由 react_agent 追加进 messages，本分支不再重复追加；只做记忆写入
    _persist_turn_memory(state, user_input, reply)
    logger.info(
        "ReAct 疗愈收尾完成 user_id={} reply_preview={!r}",
        state.get("user_id", "default_user"), preview(reply),
    )
    return {"reply": reply}


def _persist_turn_memory(state: AgentState, user_input: str, reply: str):
    """把本轮完整问答（user query + assistant reply）写入 Mem0 长期记忆。

    正常疗愈分支与危机分支共用这一条写入链路：即便命中危机干预，本轮倾诉与
    安全话术同样值得沉淀，让后续会话能感知到用户曾出现过危机信号。
    Mem0 服务端自行维护历史，提取事实时会拼接其记录的近期消息，无需客户端侧重复传入。
    """
    user_id = state.get("user_id", "default_user")
    mem0_payload = [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": reply},
    ]
    started = time.perf_counter()
    try:
        mem0_client.add(mem0_payload, user_id=user_id)
    except Exception:
        # 写入失败单独留一条带上下文的 error（含完整堆栈），再向上抛交给接口层统一转 500
        logger.exception("Mem0 记忆写入失败 user_id={} query={!r}", user_id, user_input)
        raise
    logger.info(
        "Mem0 尝试写入记忆完成 user_id={} query={!r} reply_preview={!r} elapsed={:.0f}ms",
        user_id, user_input, preview(reply), (time.perf_counter() - started) * 1000,
    )
