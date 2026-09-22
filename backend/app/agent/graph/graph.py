from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

from agent.graph.healing_response_node import healing_response_node
from agent.graph.retrieve_memory_node import retrieve_memory_node
from agent.graph.state import AgentState

# ==========================================
# 编排 LangGraph 状态图工作流
# ==========================================

# 初始化状态图，并将刚才定义的状态结构关联进去
workflow = StateGraph(AgentState)

# 将定义好的函数注册为图的“节点”
workflow.add_node("retrieve_memory", retrieve_memory_node)
workflow.add_node("generate_response", healing_response_node)

# 配置节点的连线顺序（控制执行流向）
# 流程：开始 -> 先检索记忆 -> 再根据记忆生成回复 -> 结束
workflow.add_edge(START, "retrieve_memory")
workflow.add_edge("retrieve_memory", "generate_response")
workflow.add_edge("generate_response", END)

# 编译工作流，生成可直接调用的 agent 实例
checkpointer = InMemorySaver()
soulecho_agent = workflow.compile(checkpointer=checkpointer)