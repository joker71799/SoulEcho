from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

from agent.graph.healing_response_node import healing_response_node
from agent.graph.retrieve_memory_node import retrieve_memory_node
from agent.graph.steer_direction_node import steer_direction_node
from agent.graph.state import AgentState

# ==========================================
# 编排 LangGraph 状态图工作流（含场景B：方向引导 HITL 门）
# ==========================================

# 初始化状态图，并将刚才定义的状态结构关联进去
workflow = StateGraph(AgentState)

# 将定义好的函数注册为图的“节点”
workflow.add_node("retrieve_memory", retrieve_memory_node)
workflow.add_node("steer_direction", steer_direction_node)   # HITL 暂停/恢复断点
workflow.add_node("generate_response", healing_response_node)


def route_after_retrieve(state: AgentState):
    # 会话尚未确定过陪伴方向 -> 进引导门询问；已选定 -> 直接生成，避免每轮打扰
    return "steer_direction" if not state.get("support_mode") else "generate_response"


# 配置节点的连线顺序（控制执行流向）
# 流程：开始 -> 检索记忆 -> (需要时)方向引导 -> 生成回复 -> 结束
workflow.add_edge(START, "retrieve_memory")
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