from langchain_core.messages import AIMessage

from agent.graph.state import AgentState
from agent.prompt import HEALING_PROMPT, HEALING_SYSTEM_PROMPT
from config.llm_config import chat_llm

# ==========================================
# 长期记忆组件 (Mem0 - 自托管 HTTP 服务)
# ==========================================
from config.mem0_config import mem0_client

# 写入 Mem0 时，额外带上最近多少轮历史对话作为抽取上下文。
# 单看"他今天又这样了"抽出来的事实里指代是悬空的；带上上一轮"我妈又打电话来了"，
# 抽取器才能落成"用户与母亲关系紧张"这种可检索的事实。
MEMORY_CONTEXT_TURNS = 6

# LangChain 的消息类型是 human / ai，Mem0 的 parse_messages 只认 user / assistant，
# 其余 role 会被静默丢弃，所以这里必须做一次转换。
_MEM0_ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system"}

def healing_response_node(state: AgentState):
    """
    【节点 2：疗愈对话生成与记忆同步】
    根据当前的记忆上下文以及用户的输入，生成同理心回复，并把新事件同步到记忆库。
    """
    user_input = state["messages"][-1].content  # 用户最新的心理倾诉
    memory_context = state.get("memory_context", "")  # 上一节点检索到的记忆上下文
    # 检索不到记忆时，给模型/文案一个明确的占位说明（措辞留在使用方，不污染 State）
    memory_display = memory_context if memory_context else "无历史相关记忆"

    # 调用真实大模型（阿里云百炼，经 OpenAI 兼容端点）生成疗愈回复。
    ai_message = chat_llm.invoke(HEALING_PROMPT.format(
        HEALING_SYSTEM_PROMPT=HEALING_SYSTEM_PROMPT,memory_display=memory_display,user_input=user_input))
    healing_reply = ai_message.content

    # 【核心疗愈点】：把最近几轮对话 + 本轮完整问答一起交给 Mem0 提炼事实，供下次对话使用
    recent_messages = state["messages"][-((MEMORY_CONTEXT_TURNS * 2) + 1):-1]
    mem0_client.add(
        [
            {"role": _MEM0_ROLE_MAP.get(m.type, "user"), "content": m.content}
            for m in recent_messages
        ]
        + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": healing_reply},
        ],
        user_id=state.get("user_id", "default_user"),
    )

    # 返回最终的疗愈文本，更新历史聊天记录，更新 State
    return {
        "reply": healing_reply,
        "messages": [AIMessage(content=healing_reply)]
    }