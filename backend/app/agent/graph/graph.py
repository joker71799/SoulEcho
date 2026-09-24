from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agent.graph.crisis_router_node import crisis_router_node
from agent.graph.steer_direction_node import steer_direction_node
from agent.graph.react_agent_node import react_agent_node, after_agent_router, after_tools_router
from agent.graph.react_converge_node import react_converge_node
from agent.graph.finalize_reply_node import finalize_reply_node
from agent.tools import HEALING_TOOLS
from agent.graph.state import AgentState

# ==========================================
# 编排 LangGraph 状态图工作流
# 危机优先：crisis_router 作为首节点判定，命中则直达收尾节点出危机话术，
# 跳过方向引导与 ReAct 检索（危机要的是最低延迟与受控回复，不需要历史上下文）。
# 正常链路：steer_direction 引导门（HITL）之后进入 ReAct 循环——
#   react_agent 思考并（按需）发起 search_memory 工具调用 → after_agent_router 见有 tool_calls 即去 tools 执行观察，
#   → after_tools_router 执行后才计数：未超限回到 react_agent 继续推理，
#   → 超限转 react_converge（不绑工具的强制收敛节点）产出纯文本回复再收尾，
#     杜绝消息列表全是 tool_calls 时 finalize_reply 取不到正文、reply 为空。
# 收尾单点：危机与正常两条链路都汇聚到 finalize_reply —— 它是全图唯一的 reply 写者、
#   唯一的本轮记忆沉淀者（危机出固定话术，正常取 ReAct 产物），守住“单写者”约定。
# ==========================================

# 初始化状态图，并将刚才定义的状态结构关联进去
workflow = StateGraph(AgentState)

# 将定义好的函数注册为图的“节点”
workflow.add_node("crisis_router", crisis_router_node)         # 首节点：Jev 主判，关键词仅作异常兜底
workflow.add_node("steer_direction", steer_direction_node)     # HITL 暂停/恢复断点
workflow.add_node("react_agent", react_agent_node)             # ReAct：思考/行动，自主决定是否检索记忆
workflow.add_node("react_converge", react_converge_node)       # 强制收敛：达步数上限后用不绑工具的模型产出纯文本回复
workflow.add_node("tools", ToolNode(HEALING_TOOLS))            # 工具执行引擎节点：执行工具，产出观察结果
workflow.add_node("finalize_reply", finalize_reply_node)       # 唯一收尾：定 reply（危机话术/ReAct 产物）+ 沉淀记忆


def route_after_crisis(state: AgentState):
    # 命中危机 -> 直达收尾节点（其内部走危机话术分支）；
    # 正常且本会话尚未选定陪伴方向 -> 进引导门询问；已选定 -> 跳过引导门直接进 ReAct，避免每轮打扰。
    if state.get("is_crisis"):
        return "finalize_reply"
    return "steer_direction" if not state.get("support_mode") else "react_agent"


# 配置节点的连线顺序（控制执行流向）
workflow.add_edge(START, "crisis_router")
# 危机：crisis_router -> finalize_reply；正常首轮：-> steer_direction；正常已定向：-> react_agent
workflow.add_conditional_edges(
    "crisis_router",
    route_after_crisis,
    {
        "finalize_reply": "finalize_reply",
        "steer_direction": "steer_direction",
        "react_agent": "react_agent",
    },
)
# 引导门选定方向后进入 ReAct 循环
workflow.add_edge("steer_direction", "react_agent")
# ReAct 循环：行动后先看有无工具调用（有则去执行，无则收敛），不在此计数
workflow.add_conditional_edges(
    "react_agent",
    after_agent_router,
    {"tools": "tools", "finalize_reply": "finalize_reply"},
)
# 工具执行后再计数：未超限回到 react_agent 继续推理，超限转强制收敛节点出文本
workflow.add_conditional_edges(
    "tools",
    after_tools_router,
    {"react_agent": "react_agent", "react_converge": "react_converge"},
)
# 收敛产物（纯文本 AIMessage）落进 messages 后，交由唯一收尾节点取用
workflow.add_edge("react_converge", "finalize_reply")
# 危机与正常两条链路唯一汇聚点，跑完即结束
workflow.add_edge("finalize_reply", END)

# 编译工作流，生成可直接调用的 agent 实例；checkpointer 是 HITL 暂停/恢复的关键
checkpointer = InMemorySaver()
soulecho_agent = workflow.compile(checkpointer=checkpointer)