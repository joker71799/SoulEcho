from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

from agent.graph.crisis_router_node import crisis_router_node
from agent.graph.healing_response_node import healing_response_node
from agent.graph.retrieve_memory_node import retrieve_memory_node
from agent.graph.steer_direction_node import steer_direction_node
from agent.graph.state import AgentState

# ==========================================
# 编排 LangGraph 状态图工作流
# 危机优先：crisis_router 作为首节点判定，命中则直达唯一生成节点走危机分支，
# 跳过记忆检索与方向引导（危机要的是最低延迟与受控回复，不需要历史上下文）。
# ==========================================

# 初始化状态图，并将刚才定义的状态结构关联进去
workflow = StateGraph(AgentState)

# 将定义好的函数注册为图的“节点”
workflow.add_node("crisis_router", crisis_router_node)         # 首节点：Jev 主判，关键词仅作异常兜底
workflow.add_node("retrieve_memory", retrieve_memory_node)
workflow.add_node("steer_direction", steer_direction_node)     # HITL 暂停/恢复断点
workflow.add_node("generate_response", healing_response_node)  # 唯一生成节点，内部按 is_crisis 分支


def route_after_crisis(state: AgentState):
    # 命中危机 -> 跳过检索与引导，直达生成节点（其内部走危机分支）；否则进正常检索链路
    return "generate_response" if state.get("is_crisis") else "retrieve_memory"


def route_after_retrieve(state: AgentState):
    # 会话尚未确定过陪伴方向 -> 进引导门询问；已选定 -> 直接生成，避免每轮打扰
    return "steer_direction" if not state.get("support_mode") else "generate_response"


# 配置节点的连线顺序（控制执行流向）
# 流程：开始 -> 危机识别 ->(危机)直接生成 / (正常)检索记忆 -> (需要时)方向引导 -> 生成回复 -> 结束
workflow.add_edge(START, "crisis_router")
workflow.add_conditional_edges(
    "crisis_router",
    route_after_crisis,
    {"generate_response": "generate_response", "retrieve_memory": "retrieve_memory"},
)
workflow.add_conditional_edges(
    "retrieve_memory",
    route_after_retrieve,
    {"steer_direction": "steer_direction", "generate_response": "generate_response"},
)
workflow.add_edge("steer_direction", "generate_response")
workflow.add_edge("generate_response", END)

# 编译工作流，生成可直接调用的 agent 实例；checkpointer 是 HITL 暂停/恢复的关键
checkpointer = InMemorySaver()
soulecho_agent = workflow.compile(checkpointer=checkpointer)