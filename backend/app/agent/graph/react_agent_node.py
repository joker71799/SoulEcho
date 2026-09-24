"""正常疗愈链路的 ReAct 循环：模型自主决定是否检索记忆（思考→行动→观察）。

与原 retrieve_memory→healing_response 的固定两节点相比：
- 检索这一步不再无条件执行，改由模型按需发起 search_memory 工具调用（行动）；
- 工具返回后（观察）再回到本节点，模型据此继续推理直至产出最终回复。
- 达步数上限的强制收敛拆到独立节点 react_converge_node.py（不绑工具、只能出纯文本）。
危机识别与 HITL 方向引导刻意留在本循环之外（确定性关卡，不交给模型自主决定）。
"""
from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph.state import AgentState
from agent.graph.steer_direction_node import DIRECTION_GUIDE
from agent.prompt import HEALING_SYSTEM_PROMPT_REACT
from agent.tools import HEALING_TOOLS
from config.llm_config import chat_llm
from config.logging_config import logger

# 只保留最近 N 轮原生对话喂给模型，防止长会话撑爆上下文
RECENT_REPLY_TURNS = 5
# ReAct 循环步数上限：超过强制进入收尾，杜绝模型反复调工具打转
MAX_REACT_STEPS = 3

# 绑定工具后，模型每轮要么产出 tool_calls（行动），要么产出纯文本回复（收敛）
react_llm = chat_llm.bind_tools(HEALING_TOOLS)


def build_system_content(state: AgentState) -> str:
    """组装系统提示词：疗愈基调 + 方向引导段（公开：强制收敛节点也复用同一段）。"""
    system_content = HEALING_SYSTEM_PROMPT_REACT
    # 把引导门点选的方向拼成“本次陪伴方向”追加段落（未选则不加）
    support_mode = state.get("support_mode", "")
    direction_guide = DIRECTION_GUIDE.get(support_mode, "") if support_mode else ""
    if direction_guide:
        system_content += f"\n\n【本次陪伴方向】\n{direction_guide}"
    return system_content


def react_agent_node(state: AgentState):
    """【思考/行动节点】带工具的 LLM 调用。

    产出的 AIMessage 可能含 tool_calls，由 after_agent_router 边决定去执行工具还是收尾。
    """
    # 全量传入 state["messages"]：首轮是当前倾诉；循环再入时已累积了
    # 上一轮的 tool_calls(AIMessage) 与 ToolMessage(观察结果)，模型据此继续推理。
    # 预留 MAX_REACT_STEPS*2 的余量给本轮可能产生的工具消息，避免截断掉观察结果。
    history = state["messages"][-(RECENT_REPLY_TURNS * 2 + MAX_REACT_STEPS * 2):]
    ai_message = react_llm.invoke([SystemMessage(content=build_system_content(state)), *history])
    return {"messages": [ai_message]}


def after_agent_router(state: AgentState):
    """【行动后的路由】最后一条 AIMessage 若带 tool_calls -> 去执行工具；否则收敛收尾。

    这里刻意不计数：模型既然发起了工具调用，就先让它执行拿到观察结果，
    限步判断推迟到 after_tools_router 在执行之后进行，避免被请求的工具遭丢弃。
    """
    if state["messages"][-1].tool_calls:
        return "tools"
    return "finalize_reply"


def after_tools_router(state: AgentState):
    """【观察后的路由】工具执行完再计数：未达步数上限回到 react_agent 继续推理，超限转强制收敛。

    计数放在工具执行之后，保证每一次模型请求的工具调用都真正执行并产出观察结果；
    MAX_REACT_STEPS 只限制后续还能再迭代几轮，守住延迟上限，杜绝模型反复调工具打转。
    超限不再直达 finalize_reply（彼时消息列表可能全是 tool_calls 消息、取不到正文），
    改走 react_converge 让模型基于已有观察产出纯文本收尾。
    """
    # 只统计本轮（最后一条 HumanMessage 之后）的工具轮次：
    # state["messages"] 由 checkpointer 跨轮累积，全量计数会让历史轮的工具调用
    # 吃掉当轮预算，在第 N 轮就误判超限、过早强制收敛。
    tool_rounds = 0
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            break
        if getattr(m, "tool_calls", None):
            tool_rounds += 1
    if tool_rounds >= MAX_REACT_STEPS:
        logger.warning("ReAct 达最大步数 {}，转强制收敛节点", MAX_REACT_STEPS)
        return "react_converge"
    return "react_agent"
