from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agent.graph.crisis_router_node import crisis_router_node
from agent.graph.chitchat_agent_node import chitchat_agent_node, after_chitchat_router
from agent.graph.reviewer_node import reviewer_node, after_reviewer_router
from agent.graph.case_formulation_node import (
    cf_agent_node,
    after_cf_agent_router,
    after_cf_tools_router,
)
from agent.graph.cf_converge_node import cf_converge_node
from agent.graph.supervisor_node import supervisor_node, after_supervisor_router
from agent.graph.steer_direction_node import steer_direction_node
from agent.graph.experts_node import (
    empathic_companion_node,
    socratic_clarifier_node,
    gentle_advisor_node,
    route_by_mode,
)
from agent.graph.finalize_reply_node import finalize_reply_node
from agent.tools import HEALING_TOOLS
from agent.graph.state import AgentState

# ==========================================
# 编排 LangGraph 状态图工作流
# 首节点 crisis_router 用 Jev 双判定出「危机 / 闲聊 / 正常」三路：
#   危机   ：直达 finalize_reply 出固定安全话术（最高优先，跳过一切生成与检索）；
#   闲聊   ：走 chitchat_agent 轻量单发链（不进引导门，≤一次检索）；命中记忆→Reviewer 忠实度复核，
#            纯寒暄无命中→直出收尾；
#   正常   ：CF 先行（全图唯一 ReAct 循环持有者）一手产出个案评估 + 陪伴方向 + 置信度
#            → Supervisor 按置信度路由（高置信直接按方向召专家、低置信回落引导门）
#            → 命中的专家单发生成疗愈回复（不绑工具，检索已在 CF 观察里）→ Reviewer 复核 → 收尾。
# 回炉收口（Step 10）：三专家输出无条件 → Reviewer（安全基调必查、命中记忆加查忠实度、不评方向）；
#   pass/超限降级 → finalize_reply；rewrite≤２ → 按“谁的稿谁改”回炉：闲聊回 chitchat_agent、疗愈回同一位专家；无 REV→SUP 升级线。
# CF 个案评估+分诊循环（Step 7）：cf_agent 思考并（按需）发起 search_memory
#   → 有 tool_calls 去共享 tools 执行观察、回到 cf_agent 继续；工具轮超 MAX_CF_TOOL_ROUNDS 转 cf_converge
#   用不绑工具模型强制产出结构化结果；末轮解析出 case_formulation / support_mode / support_mode_confidence。
#   同轮 CF 只完整执行一次（仅 crisis_router 能进）。
# Supervisor（Step 7 修正）：不再调 LLM，只按 CF 填好的置信度路由；高置信按 support_mode 召回三专家之一（Step 9）。
# 专家小组（Step 9）：empathic_companion / socratic_clarifier / gentle_advisor 三个单发、不绑工具节点，按方向各一段角色 prompt；引导门恢复后也按方向分派。
# 收尾单点：危机、闲聊、正常三条链路都汇聚到 finalize_reply —— 全图唯一 reply 写者、唯一记忆沉淀者。
# Reviewer（Step 5/10）：一处实现两链复用。按状态动态启用检查项；回炉按“谁的稿谁改”交回原作者（≤ MAX_REWRITE_ROUNDS，超限降级放行）。
# ==========================================

# 初始化状态图，并将刚才定义的状态结构关联进去
workflow = StateGraph(AgentState)

# 将定义好的函数注册为图的“节点”
workflow.add_node("crisis_router", crisis_router_node)         # 首节点：Jev 双判（危机+闲聊），关键词仅作异常兜底
workflow.add_node("chitchat_agent", chitchat_agent_node)       # 闲聊：单发决策、至多一次检索（非 ReAct 循环）
workflow.add_node("reviewer", reviewer_node)                   # 质量复核（一处两链复用）：闲聊命中核忠实度；疗愈链无条件过审、安全基调必查
workflow.add_node("cf_agent", cf_agent_node)                   # CF：个案评估的 ReAct 思考/行动（绑工具）
workflow.add_node("tools", ToolNode(HEALING_TOOLS))            # 共享工具节点：闲聊/CF 两链复用；执行 search_memory 后由 after_tools_router 按 is_chitchat 分派
workflow.add_node("cf_converge", cf_converge_node)             # CF 强制收敛：超工具轮上限后不绑工具强制出评估
workflow.add_node("supervisor", supervisor_node)               # 纯路由：只看 CF 填的置信度，高置信放行/低置信回落引导门（不再调 LLM）
workflow.add_node("steer_direction", steer_direction_node)     # HITL 断点（Step 8）：仅 CF 低置信时暂停，递送推断方向+理由+可覆盖选项
workflow.add_node("empathic_companion", empathic_companion_node)  # 专家·共情陪伴（listen）：单发不绑工具
workflow.add_node("socratic_clarifier", socratic_clarifier_node)  # 专家·一起理清（clarify）：单发不绑工具
workflow.add_node("gentle_advisor", gentle_advisor_node)          # 专家·温和建议（advise）：单发不绑工具
workflow.add_node("finalize_reply", finalize_reply_node)       # 唯一收尾：定 reply（危机话术/生成产物）+ 沉淀记忆


def route_after_crisis(state: AgentState):
    # 三路分岔（危机最高优先）：
    #   危机 -> finalize_reply；闲聊 -> chitchat_agent；
    #   正常 -> cf_agent（CF 先行做个案评估，每轮先跑一次；是否定调由下游 Supervisor 按沿用判断）。
    if state.get("is_crisis"):
        return "finalize_reply"
    if state.get("is_chitchat"):
        return "chitchat_agent"
    return "cf_agent"


def after_tools_router(state: AgentState):
    """共享工具节点出口（合并旧 cf_tools/chitchat_tools）：按 is_chitchat 分派回各链发起节点。

    - 闲聊链  → chitchat_agent（其自身守住 ≤1 次检索，无限步需求）；
    - CF 链    → 委托 after_cf_tools_router 做工具轮限步（未超限回 cf_agent、超限转 cf_converge）。
    工具节点本身无状态，两链共用一个实例，仅出口回向不同。
    """
    if state.get("is_chitchat"):
        return "chitchat_agent"
    return after_cf_tools_router(state)


# 配置节点的连线顺序（控制执行流向）
workflow.add_edge(START, "crisis_router")
# 三路：危机 -> finalize_reply；闲聊 -> chitchat_agent；正常 -> cf_agent
workflow.add_conditional_edges(
    "crisis_router",
    route_after_crisis,
    {
        "finalize_reply": "finalize_reply",
        "chitchat_agent": "chitchat_agent",
        "cf_agent": "cf_agent",
    },
)
# 闲聊子链：首趟带 tool_call -> 共享 tools（执行后由 after_tools_router 回 chitchat_agent）；出文本后 G1：
# retrieved_memories 非空（命中必核）-> reviewer；纯寒暄无命中 -> finalize_reply。检索上限由 chitchat_agent 自身守住。
workflow.add_conditional_edges(
    "chitchat_agent",
    after_chitchat_router,
    {
        "tools": "tools",
        "reviewer": "reviewer",
        "finalize_reply": "finalize_reply",
    },
)
# Reviewer 出口（一处两链复用）：pass/超限降级（feedback 空）-> finalize_reply；
# rewrite -> 按“谁的稿谁改”：闲聊回 chitchat_agent、疗愈按 support_mode 回同一位专家（无 SUP 升级线）
workflow.add_conditional_edges(
    "reviewer",
    after_reviewer_router,
    {
        "finalize_reply": "finalize_reply",
        "chitchat_agent": "chitchat_agent",
        "empathic_companion": "empathic_companion",
        "socratic_clarifier": "socratic_clarifier",
        "gentle_advisor": "gentle_advisor",
    },
)
# CF 子链：cf_agent 有 tool_calls -> 共享 tools 执行观察后回 cf_agent；评估已成 -> supervisor。
# tools 执行后由 after_tools_router 分派：闲聊回 chitchat_agent；CF 限步——未超限回 cf_agent、超限转 cf_converge。
workflow.add_conditional_edges(
    "cf_agent",
    after_cf_agent_router,
    {"tools": "tools", "supervisor": "supervisor"},
)
workflow.add_conditional_edges(
    "tools",
    after_tools_router,
    {"chitchat_agent": "chitchat_agent", "cf_agent": "cf_agent", "cf_converge": "cf_converge"},
)
workflow.add_edge("cf_converge", "supervisor")
# Supervisor 出口：高置信按 support_mode 召回对应专家（不打扰用户）；低置信 -> steer_direction 递送推断+理由请用户确认/覆盖
workflow.add_conditional_edges(
    "supervisor",
    after_supervisor_router,
    {
        "empathic_companion": "empathic_companion",
        "socratic_clarifier": "socratic_clarifier",
        "gentle_advisor": "gentle_advisor",
        "steer_direction": "steer_direction",
    },
)
# 引导门恢复后（用户已选定/覆盖方向）：同样按 support_mode 分派到对应专家
workflow.add_conditional_edges(
    "steer_direction",
    route_by_mode,
    {
        "empathic_companion": "empathic_companion",
        "socratic_clarifier": "socratic_clarifier",
        "gentle_advisor": "gentle_advisor",
    },
)
# 三专家产物（单发非工具 AIMessage）无条件进入 Reviewer 复核（Step 10：安全基调必查、命中记忆加查忠实度）
workflow.add_edge("empathic_companion", "reviewer")
workflow.add_edge("socratic_clarifier", "reviewer")
workflow.add_edge("gentle_advisor", "reviewer")
# 三条链路唯一汇聚点，跑完即结束
workflow.add_edge("finalize_reply", END)

# 编译工作流，生成可直接调用的 agent 实例；checkpointer 是 HITL 暂停/恢复的关键
checkpointer = InMemorySaver()
soulecho_agent = workflow.compile(checkpointer=checkpointer)