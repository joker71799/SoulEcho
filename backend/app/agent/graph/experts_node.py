"""专家小组三节点（Step 9）：单发生成、不绑工具，按 support_mode 召其一。

三者结构一致、只有"陪伴方式"的角色化 system prompt 不同（基座 EXPERT_BASE + 各自角色段，均集中在 prompt.py），
语气/策略差异明显：
- empathic_companion（listen） ：共情陪伴，接住情绪，不分析不建议；
- socratic_clarifier（clarify）：一起理清，复述+开放式提问，陪用户自己看清；
- gentle_advisor （advise ）   ：温和建议，先共情再给少量可操作的小建议。

检索已在 CF 阶段完成、观察留在 messages 里，故三专家一律不绑工具（全图唯一 ReAct 循环已归 CF）；
生成时引用 CF 的 case_formulation 作贴合参考。产物是本轮最后一条非工具 AIMessage，
先交 Reviewer 复核（Step 10）、再由 finalize_reply 取作 reply。
"""
from langchain_core.messages import SystemMessage

from agent.graph.case_formulation_node import MAX_CF_TOOL_ROUNDS
from agent.graph.state import AgentState, RECENT_REPLY_TURNS
from agent.prompt import ADVISOR_ROLE, EMPATHIC_ROLE, EXPERT_BASE, SOCRATIC_ROLE
from config.llm_config import chat_llm
from config.logging_config import logger, preview

# 专家单发生成：刻意不绑工具，检索归 CF，只能基于上下文已有观察出文本
expert_llm = chat_llm

# 陪伴方向 → 专家节点名（单一映射；Supervisor 高置信与引导门恢复后都按它分派）
EXPERT_BY_MODE = {
    "listen": "empathic_companion",
    "clarify": "socratic_clarifier",
    "advise": "gentle_advisor",
}
_DEFAULT_EXPERT = "empathic_companion"  # 方向缺失/异常时的安全兜底（先接住情绪）


def _case_block(state: AgentState) -> str:
    """把 CF 个案评估拼成参考段（无评估则不加，避免空段干扰）。"""
    case = state.get("case_formulation") or ""
    return f"\n\n【个案评估参考（CF 产出，供你贴合用户处境）】\n{case}" if case else ""


def _rewrite_block(state: AgentState) -> str:
    """回炉定向重稿指令（Step 10）：Reviewer 判 rewrite 时拼上，锁死陪伴方向与事实、只按意见修订措辞。

    无 review_feedback（首次生成草稿）时返回空串，与本模块外的旧行为一致；
    上一版草稿已在 messages 里（Reviewer 不往 messages 写东西），故模型能看到旧稿并据此修订。
    """
    feedback = (state.get("review_feedback") or "").strip()
    if not feedback:
        return ""
    return (
        "\n\n【回炉定向重稿】上一版草稿经质检需修改。请在保持【陪伴方向不变】、"
        "不新增未经检索的事实的前提下，只针对下面的意见修订后重新给出本轮回复：\n" + feedback
    )


def _make_expert_node(expert_name: str, role_prompt: str):
    """工厂：把角色 prompt 绑定成一个单发、不绑工具的专家节点函数。"""
    system_content = EXPERT_BASE + role_prompt

    def expert_node(state: AgentState):
        user_id = state.get("user_id", "default_user")
        conversation_id = state.get("conversation_id", "")
        # 窗口留足 CF 本轮的工具轮与观察，确保专家能看到 CF 检索到的记忆
        history = state["messages"][-(RECENT_REPLY_TURNS * 2 + MAX_CF_TOOL_ROUNDS * 3):]
        ai_message = expert_llm.invoke(
            [SystemMessage(content=system_content + _case_block(state) + _rewrite_block(state)), *history]
        )
        logger.info(
            "专家生成 user_id={} conversation_id={} expert={} reply_preview={!r}",
            user_id, conversation_id, expert_name, preview(str(ai_message.content)),
        )
        return {"messages": [ai_message]}

    return expert_node


empathic_companion_node = _make_expert_node("empathic_companion", EMPATHIC_ROLE)
socratic_clarifier_node = _make_expert_node("socratic_clarifier", SOCRATIC_ROLE)
gentle_advisor_node = _make_expert_node("gentle_advisor", ADVISOR_ROLE)


def route_by_mode(state: AgentState) -> str:
    """按本轮 support_mode 召回对应专家；方向缺失/非法兜底为共情陪伴（先接住情绪）。"""
    return EXPERT_BY_MODE.get(state.get("support_mode") or "", _DEFAULT_EXPERT)
