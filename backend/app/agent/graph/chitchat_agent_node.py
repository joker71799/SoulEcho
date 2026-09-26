"""闲聊 Agent 节点（Step 3）：单发决策的轻量链路，非 ReAct 循环。

与主链的三路分工（危机 / 闲聊 / 正常）：
- 危机     → finalize 固定模板（最高优先，冲突时压掉闲聊）；
- 正常倾诉 → HITL 引导门 + ReAct 检索生成；
- 纯寒暄   → 本节点。

本节点只绑 search_memory 做「至多一次」的记忆核验，全程不开放循环：
- 首趟（绑工具）：无 tool_call → 直接出文本（仅 1 次 LLM 调用）；
- 若首趟发起 tool_call → 共享 tools 节点执行观察 → 回到本节点；
- 第二趟改用不绑工具的 LLM，结构上无法再产 tool_call，只能基于已有观察出终稿。
故整条闲聊链 ≤2 次 LLM 调用、固定最多检索一次。

闲聊轮不进 HITL 引导门；本步暂不经 Reviewer（Step 5 再接线做忠实度复核）。
"""
from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph.state import AgentState, RECENT_REPLY_TURNS
from agent.prompt import CHITCHAT_SYSTEM_PROMPT, REWRITE_DIRECTIVE
from agent.tools import HEALING_TOOLS
from config.llm_config import chat_llm
from config.logging_config import logger, preview

# 首趟：绑工具，把「这轮要不要核验记忆」交给模型单发决定
chitchat_llm = chat_llm.bind_tools(HEALING_TOOLS)
# 终稿趟：检索后强制出文本，刻意不绑工具，从结构上杜绝再次发起 tool_call（守住 ≤1 次检索）
chitchat_final_llm = chat_llm


def _retrieved_this_turn(state: AgentState) -> bool:
    """本轮（最后一条 HumanMessage 之后）是否已经跑过一次检索。

    穿过工具消息反向找：先遇到 HumanMessage 说明本轮还没检索过；遇到带 tool_calls 的
    AIMessage 说明首趟已发起检索、工具观察已回到消息列表，本趟应出终稿。
    """
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return False
        if getattr(m, "tool_calls", None):
            return True
    return False


def chitchat_agent_node(state: AgentState):
    """【闲聊生成节点】单发决策、至多一次检索；被 Reviewer 回炉时携意见定向重稿。"""
    user_id = state.get("user_id", "default_user")
    conversation_id = state.get("conversation_id", "")
    already_retrieved = _retrieved_this_turn(state)
    # 已检索过走不绑工具的终稿趟（含回炉重稿），否则走绑工具的首趟
    llm = chitchat_final_llm if already_retrieved else chitchat_llm

    system_content = CHITCHAT_SYSTEM_PROMPT
    feedback = state.get("review_feedback") or ""
    is_rewrite = already_retrieved and bool(feedback)
    if is_rewrite:
        system_content += REWRITE_DIRECTIVE.format(feedback=feedback)

    # 窗口留出本轮工具调用与观察的余量，确保终稿趟能看到已检索到的记忆
    history = state["messages"][-(RECENT_REPLY_TURNS * 2 + 2):]
    ai_message = llm.invoke([SystemMessage(content=system_content), *history])

    if is_rewrite:
        logger.info(
            "闲聊回炉定向重稿 user_id={} conversation_id={} rewrite_count={} reply_preview={!r}",
            user_id, conversation_id, state.get("rewrite_count", 0), preview(str(ai_message.content)),
        )
    elif already_retrieved:
        logger.info(
            "闲聊终稿生成（已检索一次）user_id={} conversation_id={} reply_preview={!r}",
            user_id, conversation_id, preview(str(ai_message.content)),
        )
    else:
        logger.info(
            "闲聊首判 user_id={} conversation_id={} will_retrieve={} content_preview={!r}",
            user_id, conversation_id, bool(ai_message.tool_calls), preview(str(ai_message.content)),
        )
    return {"messages": [ai_message]}


def after_chitchat_router(state: AgentState):
    """首趟带 tool_call → 去执行核验检索；已出文本则走 G1 条件边：

    retrieved_memories 非空（命中必核）→ reviewer；纯寒暄无命中 → 直达 finalize_reply。
    """
    if state["messages"][-1].tool_calls:
        return "tools"
    if state.get("retrieved_memories"):
        return "reviewer"
    return "finalize_reply"
