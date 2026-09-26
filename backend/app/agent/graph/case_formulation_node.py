"""CF Agent：正常链每轮的"大脑"（Step 7 修正版）——全图唯一 ReAct 循环持有者。

职责（按用户澄清修正）：CF 一手产出三样东西，Supervisor 不再判断方向、只做纯路由。
- 绑 search_memory 按需多步检索（≤ MAX_CF_TOOL_ROUNDS）；末轮（无 tool_calls）产出一个 JSON：
    {"case_formulation": 个案评估, "support_mode": listen|clarify|advise,
     "confidence": 0~1, "reason": "一句话理由"}
  一次性写回三个状态位：case_formulation / support_mode / support_mode_confidence。
- support_mode 每轮由 CF 重新推断填充（crisis_router 每轮清空），不再会话级沿用；
- 解析失败或方向非法一律降级：case_formulation 兜住原文、support_mode 不写、confidence=0.0，
  交由 Supervisor 走低置信回落引导门，绝不卡住主链。

检索触发原则（写进 prompt）：涉往事 / 指代不明才检索，纯当下情绪事件零检索直出——
多数轮次一次调用即出结构化结果（低延迟），只有"像上次那样"类句子才驱动多步检索。

react_agent 的旧 ReAct 机制迁移到此（绑工具循环 + 执行后限步 + 超限强制收敛），仅换评估/分诊人设。
超限强制收敛节点 cf_converge_node 及其 cf_converge_llm 拆到同目录 cf_converge_node.py，复用本模块的 cf_window / cf_update。
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph.state import AgentState, RECENT_REPLY_TURNS
from agent.graph.steer_direction_node import DIRECTION_OPTIONS
from agent.prompt import CF_SYSTEM_PROMPT
from agent.tools import HEALING_TOOLS
from config.llm_config import chat_llm
from config.logging_config import logger

# CF 工具轮上限：比疗愈 ReAct 更紧（评估不需要反复翻历史），守住延迟
MAX_CF_TOOL_ROUNDS = 2

# 合法方向集合复用引导门的单一数据源，避免两处各写一份漂移
_VALID_MODES = {o["value"] for o in DIRECTION_OPTIONS}

# 循环趟：绑 search_memory，让模型自判「这轮要不要翻历史」。
# CF 是分诊判定任务（选 support_mode + 给 confidence + 吐结构化 JSON），刻意压到低温求稳定可复现（与 Reviewer 同约定）。
# 收敛趟的 cf_converge_llm 随强制收敛节点一起搬到 cf_converge_node.py。
cf_llm = chat_llm.bind_tools(HEALING_TOOLS, temperature=0.0)


def cf_window(state: AgentState) -> list:
    """CF 与强制收敛共用的消息窗口：留足本轮工具轮与观察的余量。"""
    return state["messages"][-(RECENT_REPLY_TURNS * 2 + MAX_CF_TOOL_ROUNDS * 3):]


def _cf_tool_rounds(state: AgentState) -> int:
    """统计本轮（最后一条 HumanMessage 之后）CF 已发起的工具轮次。

    只数本轮，避免 checkpointer 跨轮累积的历史工具调用吃掉当轮预算导致过早收敛。
    """
    rounds = 0
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            break
        if getattr(m, "tool_calls", None):
            rounds += 1
    return rounds


def _parse_cf_output(raw: str):
    """稳健解析 CF 末轮输出为 (case_formulation, support_mode|None, confidence, reason)。

    非法模式 / 解析失败一律降级：case_formulation 兜住原文、support_mode=None、confidence=0.0，
    交给 Supervisor 走低置信回落引导门，绝不因 CF 输出不规范而卡住主链。
    """
    text = str(raw).strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start : end + 1]) if start != -1 and end != -1 else None
    except (ValueError, TypeError):
        data = None
    if not isinstance(data, dict):
        logger.warning("CF 输出无法解析为 JSON，降级为低置信回落引导门 raw={!r}", text)
        return text, None, 0.0, ""

    case = str(data.get("case_formulation") or "").strip() or text
    mode = data.get("support_mode")
    mode = mode if mode in _VALID_MODES else None
    try:
        confidence = float(data.get("confidence", 0.0))
    except (ValueError, TypeError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)
    # 方向非法则置信度失去意义：一并压到 0，强制走引导门
    if mode is None:
        confidence = 0.0
    reason = str(data.get("reason") or "")
    return case, mode, confidence, reason


def cf_update(ai_message):
    """把 CF 末轮消息解析为状态更新：始终写 case_formulation / support_mode / 置信度 / 理由。

    非法方向记为空串（与 crisis_router 每轮重置的「无方向」哨兵一致），不把 None 落进 str 字段。
    """
    case, mode, confidence, reason = _parse_cf_output(str(ai_message.content))
    update = {
        "messages": [ai_message],
        "case_formulation": case,
        # 非法方向归空串（=每轮重置的「无方向」哨兵），不让 None 污染 str 字段
        "support_mode": mode or "",
        "support_mode_confidence": confidence,
        # 推断理由无条件落态：高置信直通时只进日志，低置信时由引导门递送供用户参考/覆盖
        "support_mode_reason": reason,
    }
    logger.info(
        "CF 评估+分诊完成 case_len={} support_mode={} confidence={:.2f} reason={!r}",
        len(case), mode, confidence, reason[:60],
    )
    return update


def cf_agent_node(state: AgentState):
    """【CF 思考/行动节点】带工具的 LLM 调用；无 tool_calls 即视为评估已成，解析并写回三态。"""
    ai_message = cf_llm.invoke(
        [SystemMessage(content=CF_SYSTEM_PROMPT), *cf_window(state)]
    )
    if ai_message.tool_calls:
        # 还要继续检索：只把带 tool_calls 的 AIMessage 落进 messages，交 cf_tools 执行
        logger.info("CF 发起检索 tool_calls={}", len(ai_message.tool_calls))
        return {"messages": [ai_message]}
    return cf_update(ai_message)


def after_cf_agent_router(state: AgentState):
    """CF 步后路由：带 tool_calls → 去共享 tools 节点执行观察；否则结果已成 → 交 Supervisor 路由。"""
    if state["messages"][-1].tool_calls:
        return "tools"
    return "supervisor"


def after_cf_tools_router(state: AgentState):
    """cf_tools 执行后限步：未超限回 cf_agent 继续推理，超限转 cf_converge 强制出结果。"""
    if _cf_tool_rounds(state) >= MAX_CF_TOOL_ROUNDS:
        return "cf_converge"
    return "cf_agent"
