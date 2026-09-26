"""Reviewer 质量复核节点（Step 4：先把节点做扎实，暂不接线，Step 5/10 再接入图）。

单一实现、按状态动态启用检查项（一次低温 LLM 调用给结论）：
- retrieved_memories 非空  → 启用【忠实度】：回复涉及用户过去的陈述须与检索记忆一致、不得编造；
- 疗愈链回复（非闲聊）     → 【安全基调】必查：接纳共情、不说教、不含危险/伤害性引导。

职责边界（硬约束，写进 prompt 与注释）：
- Reviewer 只评「本轮待发送草稿」的表面质量（安全基调 / 忠实度），
  **绝不做方向判断**——方向对错归上游 CF 评估 + Supervisor 诊断 + HITL 三道防线；
- verdict 仅 pass / rewrite 两态，rewrite 附一条具体可执行的修改意见文本。

回炉纪律：rewrite_count 记本轮已重写次数，最多回炉 MAX_REWRITE_ROUNDS(=2) 次；
超限则取最后一版**降级放行**（feedback 置空，让后续条件边走 FIN）。
危机固定模板、纯寒暄不经过本节点（由接线方 graph 的边保证，见 Step 5/10）。
"""
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.graph.experts_node import route_by_mode
from agent.graph.state import AgentState
from agent.prompt import REVIEWER_SYSTEM_PROMPT
from config.llm_config import chat_llm
from config.logging_config import logger, preview

# 回炉上限：Reviewer 最多要求重写 2 次，超限取最后一版降级放行，杜绝反复打转拉高延迟
MAX_REWRITE_ROUNDS = 2

# 复核是判定任务，追求稳定可复现，刻意压到低温（不套用生成侧 AGENT_LLM_TEMPERATURE 的高温）
reviewer_llm = chat_llm.bind(temperature=0.0)

# 疗愈链安全基调：所有疗愈回复必查（危机模板/纯寒暄不经本节点，由接线方保证）
_SAFETY_CHECK = (
    "【安全基调】草稿须符合心理疗愈的安全基调：接纳、共情、不说教不评判、语气温暖自然，"
    "不含任何引导自我伤害或危险的建议；违反则判 rewrite。"
)


def _faithfulness_check(retrieved_memories: list) -> str:
    """忠实度检查项：把本轮过阈值的检索记忆作为「过去的全部事实」递给复核模型。"""
    lines = "\n".join(f"- {m}" for m in retrieved_memories)
    return (
        "【忠实度】草稿中凡涉及用户过去的陈述，必须与下面这份检索记忆一致，"
        "且不得凭空多出记忆里没有的具体细节（人名、地点、时间、偏好、经历等）；"
        "把没有的说成有、或与记忆矛盾，判 rewrite。\n"
        f"检索到的历史记忆（关于用户过去的全部事实）：\n{lines}"
    )


def _current_draft(state: AgentState) -> str:
    """反向取最后一条「有正文、非工具调用」的 AIMessage 作为待审草稿（与 finalize 取稿口径一致）。"""
    return next(
        (
            m.content
            for m in reversed(state["messages"])
            if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None)
        ),
        "",
    )


def _parse_verdict(raw: str):
    """从模型输出稳健地抠出 (verdict, feedback)；解析失败按 pass 放行，绝不因复核异常卡住用户。"""
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start : end + 1]) if start != -1 and end != -1 else {}
        verdict = str(data.get("verdict", "")).lower()
        feedback = str(data.get("feedback") or "")
    except (ValueError, TypeError):
        logger.warning("Reviewer 输出无法解析为 JSON，按 pass 放行 raw={!r}", raw)
        return "pass", ""
    return (verdict if verdict in ("pass", "rewrite") else "pass"), feedback


def reviewer_node(state: AgentState):
    """【质量复核节点】按状态动态启用检查项，输出 rewrite_count / review_feedback 供后续条件边消费。"""
    user_id = state.get("user_id", "default_user")
    conversation_id = state.get("conversation_id", "")
    draft = _current_draft(state)
    is_chitchat = bool(state.get("is_chitchat"))
    retrieved = state.get("retrieved_memories") or []

    # 动态拼检查项：闲聊命中核验只查忠实度；疗愈链（非闲聊）安全基调必查，命中记忆再加忠实度
    checks = []
    if retrieved:
        checks.append(_faithfulness_check(retrieved))
    if not is_chitchat:
        checks.append(_SAFETY_CHECK)
    if not checks:  # 兜底：接线方应保证不至此，万一至此仍按安全基调审，避免空标准放行
        checks.append(_SAFETY_CHECK)

    system_content = REVIEWER_SYSTEM_PROMPT.format(checks="\n".join(f"- {c}" for c in checks))
    review_messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=f"【待审草稿】\n{draft}"),
    ]

    raw = reviewer_llm.invoke(review_messages)
    verdict, feedback = _parse_verdict(str(raw.content))

    rewrite_count = state.get("rewrite_count", 0)
    logger.info(
        "Reviewer 复核完成 user_id={} conversation_id={} verdict={} rewrite_count={}/{} retrieved={} is_chitchat={}",
        user_id, conversation_id, verdict, rewrite_count, MAX_REWRITE_ROUNDS, len(retrieved), is_chitchat,
    )

    # 判 rewrite 且未超限：计数 +1，写回意见，交给回炉边（Step 5/10）
    if verdict == "rewrite" and rewrite_count < MAX_REWRITE_ROUNDS:
        return {"rewrite_count": rewrite_count + 1, "review_feedback": feedback}
    # rewrite 但已达上限：降级放行——取最后一版，清空意见让条件边走 FIN
    if verdict == "rewrite":
        logger.warning(
            "Reviewer 回炉已达上限 {}，取最后一版降级放行 user_id={} conversation_id={}",
            MAX_REWRITE_ROUNDS, user_id, conversation_id,
        )
    # pass / 超限降级：feedback 置空 = 放行
    return {"rewrite_count": rewrite_count, "review_feedback": ""}


def after_reviewer_router(state: AgentState):
    """Reviewer 出口路由（Step 10 全图收口）：review_feedback 空（pass/降级）→收尾；
    非空（rewrite）→按“谁的稿谁改”交回原作者定向重稿：
    - 闲聊链 → chitchat_agent；
    - 疗愈链 → 按本轮 support_mode 召回的同一位专家（route_by_mode）。
    不存在 REV→SUP 升级线：方向对错归 CF 评估 + Supervisor 诊断 + HITL 三道防线。
    """
    if not state.get("review_feedback"):
        return "finalize_reply"
    if state.get("is_chitchat"):
        return "chitchat_agent"
    return route_by_mode(state)
