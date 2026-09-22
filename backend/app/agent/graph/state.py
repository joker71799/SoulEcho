from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # messages 存储对话历史，Annotated + add_messages 告诉 LangGraph：
    # 当有新消息进来时，不是覆盖旧数据，而是追加（Append）到列表末尾。
    messages: Annotated[list, add_messages]

    # 用户的唯一标识符，用来隔离和检索不同用户的疗愈记忆
    user_id: str

    # 从 Mem0 检索出的历史记忆上下文（纯数据，不含任何提示语/装饰文本）
    # 由 retrieve_memory_node 写入，供 healing_response_node 拼进系统提示词的【历史记忆】段落
    memory_context: str

    # 最终返回给前端的疗愈回复文本，只由 generate_response 节点写入
    reply: str