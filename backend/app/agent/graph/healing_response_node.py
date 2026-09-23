import time

from langchain_core.messages import AIMessage, SystemMessage

from agent.graph.state import AgentState
from agent.prompt import HEALING_SYSTEM_PROMPT
from agent.graph.steer_direction_node import DIRECTION_GUIDE
from config.llm_config import chat_llm
from config.logging_config import logger, preview

# ==========================================
# 长期记忆组件 (Mem0 - 自托管 HTTP 服务)
# ==========================================
from config.mem0_config import mem0_client

# 调用 LLM 时，本会话内原样传入的近期对话轮数（不含当前这轮），
# 让模型以原生多轮方式理解会话，而不靠把历史拼进一段文本。
RECENT_REPLY_TURNS = 5

# 拼进系统提示词的记忆段落模板（跨会话长期记忆仅供理解背景，不必逐条复述）。
# 只是本节点的消息拼装措辞，不进 prompt.py，那里只留真正的提示词。
MEMORY_SECTION = (
    "【历史记忆】（来自跨会话长期记忆，仅供理解用户背景，不必逐条复述）\n{memory_display}"
)


def healing_response_node(state: AgentState):
    """
    【节点 2：疗愈对话生成与记忆同步】
    根据当前的记忆上下文以及用户的输入，生成同理心回复，并把新事件同步到记忆库。
    """
    user_input = state["messages"][-1].content  # 用户最新的心理倾诉
    memory_context = state.get("memory_context", "")  # 上一节点检索到的记忆上下文
    # 检索到记忆就拼成【历史记忆】段落，检索不到填空串，段落整体消失不留占位文本
    memory_section = (
        MEMORY_SECTION.format(memory_display=memory_context) if memory_context else ""
    )

    # 场景B：把用户在引导门点选的方向拼成“本次陪伴方向”追加段落（未选则不加）
    support_mode = state.get("support_mode", "")
    direction_guide = DIRECTION_GUIDE.get(support_mode, "") if support_mode else ""
    system_content = HEALING_SYSTEM_PROMPT.format(memory_section=memory_section)
    if direction_guide:
        system_content += f"\n\n【本次陪伴方向】\n{direction_guide}"

    # 多轮 messages 调用：单条 System（角色设定 + 跨会话记忆合并）在前，
    # 本会话近期历史以 HumanMessage / AIMessage 原样传入，最后是当前倾诉，
    # 让模型以原生多轮方式理解会话，而不是把一切压成一段字符串。
    # 只取最近 RECENT_REPLY_TURNS 轮，防止长会话把上下文窗口撑爆。
    history = state["messages"][-(RECENT_REPLY_TURNS * 2):-1]
    llm_messages = [
        SystemMessage(content=system_content)
    ]
    llm_messages.extend(history)
    llm_messages.append(state["messages"][-1])

    # 调用真实大模型（阿里云百炼，经 OpenAI 兼容端点）生成疗愈回复（计时观察生成耗时）。
    started = time.perf_counter()
    ai_message = chat_llm.invoke(llm_messages)
    healing_reply = ai_message.content
    logger.info(
        "疗愈回复生成完成 user_id={} memory_hit={} reply_preview={!r} elapsed={:.0f}ms",
        state.get("user_id", "default_user"), bool(memory_context),
        preview(healing_reply), (time.perf_counter() - started) * 1000,
    )

    # 【记忆写入】：只传本轮完整问答（user query + assistant reply），
    # Mem0 服务端自行维护历史，提取事实时会拼接其记录的近期消息，无需客户端侧重复传入。
    mem0_payload = [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": healing_reply},
    ]
    started = time.perf_counter()
    try:
        mem0_client.add(
            mem0_payload,
            user_id=state.get("user_id", "default_user"),
        )
    except Exception:
        # 写入失败单独留一条带上下文的 error（含完整堆栈），再向上抛交给接口层统一转 500
        logger.exception(
            "Mem0 记忆写入失败 user_id={} query={!r}",
            state.get("user_id", "default_user"), user_input,
        )
        raise
    logger.info(
        "Mem0 记忆写入完成 user_id={} query={!r} reply_preview={!r} elapsed={:.0f}ms",
        state.get("user_id", "default_user"), user_input, preview(healing_reply),
        (time.perf_counter() - started) * 1000,
    )

    # 返回最终的疗愈文本，更新历史聊天记录，更新 State
    return {
        "reply": healing_reply,
        "messages": [AIMessage(content=healing_reply)]
    }