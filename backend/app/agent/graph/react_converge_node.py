"""强制收敛节点：ReAct 达步数上限后专责产出纯文本最终回复（方案B）。

达上限时消息列表可能全是带 tool_calls 的 AIMessage，finalize_reply 反向筛选
取不到正文会得到空 reply。本节点用不绑工具的 LLM 再调一次——请求里没有
tools schema，模型结构上无法再产出 tool_calls，只能基于已有工具观察直接给出
疗愈回复，从根上保证 finalize_reply 一定能取到非空正文（比固定话术兜底质量高）。
"""
from langchain_core.messages import SystemMessage

from agent.graph.state import AgentState
from agent.graph.react_agent_node import (
    MAX_REACT_STEPS,
    RECENT_REPLY_TURNS,
    build_system_content,
)
from agent.prompt import CONVERGE_DIRECTIVE
from config.llm_config import chat_llm
from config.logging_config import logger

# 强制收敛专用：刻意不 bind_tools，从结构上杜绝再产出 tool_calls，只能输出纯文本
converge_llm = chat_llm


def react_converge_node(state: AgentState):
    """【强制收敛节点】达步数上限后专责产出纯文本最终回复。

    上一步工具观察结果都已在消息历史里；用不绑工具的 converge_llm 调用，
    模型无法再发起 tool_calls，只能基于已有信息直接给出疗愈回复。
    """
    # 与 react_agent 相同的截断窗口：本轮的工具调用与观察结果都必须在窗口内，
    # 否则收敛时模型看不到已检索到的记忆，回复会失去检索的意义。
    history = state["messages"][-(RECENT_REPLY_TURNS * 2 + MAX_REACT_STEPS * 2):]
    system_content = f"{build_system_content(state)}\n\n{CONVERGE_DIRECTIVE}"
    ai_message = converge_llm.invoke([SystemMessage(content=system_content), *history])
    logger.info(
        "ReAct 强制收敛生成最终回复 user_id={} content_preview={!r}",
        state.get("user_id", "default_user"),
        (ai_message.content[:80] if ai_message.content else ""),
    )
    return {"messages": [ai_message]}
