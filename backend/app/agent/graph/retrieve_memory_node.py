import time

from agent.graph.state import AgentState
from agent.prompt import QUERY_REWRITE_PROMPT
from config.llm_config import chat_llm
from config.logging_config import logger

# ==========================================
# 长期记忆组件 (Mem0 - 自托管 HTTP 服务)
# ==========================================
# 已放弃进程内嵌入的 Memory.from_config 模式，统一通过 mem0_client 以 HTTP 方式
# 调用独立部署的自托管 Mem0 服务。配置与连接细节见 backend/app/config/mem0_config.py。
# Mem0 连接配置由 config.mem0_config 从 .env 加载，这里直接复用已建好的 HTTP 客户端单例
from config.mem0_config import mem0_client

# query 改写是确定性转换任务，用低温度避免模型自由发挥、引入噪声。
rewrite_llm = chat_llm.bind(temperature=0)

# 检索 Mem0 时，拼接最近多少轮对话作为 query，
# 解决用户当前输入指代模糊（"还是那件事""他又来了"）导致向量召回失败的问题。
SEARCH_RECENT_TURNS = 5

def retrieve_memory_node(state: AgentState):
    """
    【节点 1：记忆检索】
    在处理这轮对话的输入前，系统会基于当前用户query，前置检索该用户在 Mem0
    长期记忆库中沉淀的历史背景与情感画像，为后续生成连续、深度的疗愈回复提供上下文。
    """
    user_id = state.get("user_id", "default_user")
    messages = state["messages"]
    # 当前用户最新倾诉，作为待改写的原始 query
    current_query = messages[-1].content
    # 近 N 轮历史（不含当前这一句），作为 query 改写的上下文锚点
    history_messages = messages[-(SEARCH_RECENT_TURNS * 2 + 1):-1]

    # 先做 query 改写：把指代模糊的当前倾诉结合历史，改写成可独立检索的完整查询。
    # 首轮无历史时跳过改写，省一次 LLM 调用，直接用当前倾诉检索。
    if history_messages:
        history_text = "\n".join([m.content for m in history_messages])
        rewrite_prompt = QUERY_REWRITE_PROMPT.format(
            history=history_text, current=current_query
        )
        rewritten_query = rewrite_llm.invoke(rewrite_prompt).content.strip()
        # 改写为空时兜底回退到原始倾诉，避免检索拿到空串
        if not rewritten_query:
            logger.warning(
                "query 改写结果为空，回退原始倾诉 user_id={} current_query={!r}",
                user_id, current_query,
            )
        search_query = rewritten_query or current_query
        # 记录改写前后对照，方便观察 query 改写质量、调优提示词
        logger.info(
            "query 改写完成 user_id={} original={!r} rewritten={!r}",
            user_id, current_query, search_query,
        )
    else:
        search_query = current_query
        logger.info("首轮对话跳过 query 改写 user_id={} query={!r}", user_id, current_query)

    # 从记忆库检索与当前话题相关的历史记忆（计时，慢查询往往是链路第一瓶颈）
    started = time.perf_counter()
    previous_memories = mem0_client.search(search_query, user_id=user_id)
    search_ms = (time.perf_counter() - started) * 1000

    # 将检索到的散落记忆碎片，拼接成一段上下文文本
    memory_context = ""
    if previous_memories and "results" in previous_memories:
        memory_context = "\n".join([m["memory"] for m in previous_memories["results"]])
        hit_count = len(previous_memories["results"])
    else:
        hit_count = 0
    logger.info(
        "记忆检索完成 user_id={} hit_count={} elapsed={:.0f}ms",
        user_id, hit_count, search_ms,
    )

    # 把检索结果单独写进 memory_context 字段，保持字段语义单一：
    # 这里只负责“找到记忆”，不负责“怎么措辞给模型看”
    return {"memory_context": memory_context}