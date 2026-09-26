from langchain_core.messages import AIMessage, HumanMessage
from langchain_typesafe import Choice

from agent.crisis.keyword_scan import scan_crisis
from agent.graph.state import AgentState
from config.logging_config import logger, preview
from config.typesafe_config import jev_classifier

# ==========================================
# 危机识别首节点：Jev 为主判，关键词仅在 Jev 异常时兜底
# ==========================================
# 为什么用 Jev 而不是通用大模型做意图识别：
# Jev 是 System One 决策模型，不生成文本，直接返回带校准概率的结构化判断，
# 延迟与成本都远低于大模型，天生就是为「分类/路由」这类判断任务设计的。
# 策略说明（经确认采用）：Jev 调用成功时一律以它的结论为准，关键词层不参与；
# 仅当 Jev 调用失败才回落关键词。
#
# Step 1 升级（纯增量）：同一次 Jev invoke 里同时拿回「危机等级 + 是否闲聊」双判。
# 本轮只多写一个 is_chitchat 状态位并记日志，路由行为保持原样（闲聊仍走旧链路）。
#
# 送判文本升级：危机信号可能藏在指代里（「我明天就要去找她了」/「那瓶东西吃完了」），
# 裸句必漏判。故不单独送本轮 query，而是拼接最近 2 轮原话作为上下文（零 LLM 改写、
# 不丢用户本意），并在 instructions 里声明结构：历史仅用于补全指代，判定落点仍是当前 query。
CRISIS_LEVEL_QUESTION = Choice(
    instructions=(
        "输入包含「历史聊天记录」与「当前用户query」两部分。"
        "请判断用户当前这句话属于哪种自伤/自杀危机等级："
        "历史仅用于补全当前句中的指代与省略（如「她」「那瓶东西」指向谁/什么），"
        "判定落点始终是当前用户query 本身。"
    ),
    criteria={
        "none": (
            "没有自伤或自杀念头。普通的情绪低落、疲惫、压力大、失眠、"
            "抱怨生活等都归这一类。"
        ),
        "risk": (
            "有风险倾向：出现伤害自己的念头或暗示，但没有明确计划、手段或即刻时间窗。"
        ),
        "imminent": (
            "即刻危险：表达了明确的自杀念头、计划、手段，或暗示马上就要实施。"
        ),
    },
)

# 闲聊判定：与危机等级在同一次 Jev invoke 中一起拿回，共享同一次调用的延迟。
# 本轮仅用于写状态 + 日志，尚不参与路由（真正分岔在 Step 3 接入）。
CHITCHAT_QUESTION = Choice(
    instructions=(
        "输入包含「历史聊天记录」与「当前用户query」两部分。"
        "请判断用户当前这句话是否只是闲聊/寒暄（不涉及需要陪伴引导的情绪倾诉）："
        "历史仅用于补全指代与省略，判定落点始终是当前用户query 本身。"
    ),
    criteria={
        "yes": "纯粹的打招呼、寒暄、日常闲聊，与心理困扰、情绪倾诉无关。",
        "no": "包含情绪倾诉、压力困扰、需要被陪伴或引导的内容；或任何危机信号。",
    },
)

# Jev 会返回校准过的置信度，官方建议「风险容忍度由调用方代码显式声明」：
# 低置信度时不把等级升级到最警戒的 imminent 话术，避免过度惊吓用户；
# 但仍按 risk 出卡，保证宁可多报也不漏报。
_IMMINENT_MIN_CONFIDENCE = 0.6

_LEVEL_RANK = {"none": 0, "risk": 1, "imminent": 2}


def crisis_router_node(state: AgentState):
    """
    【首节点：危机识别 + 闲聊双判】
    以 Jev 的结构化判定为唯一决策来源；仅当 Jev 调用异常（网络/额度/超时等）时，
    才回落确定性关键词扫描，保证 Jev 不可用时危机识别不至于完全失效。
    命中危机的轮次由下游路由直接跳到唯一生成节点走危机分支，跳过检索与方向引导。
    Step 1 增量：同一次 Jev invoke 额外拿回 is_chitchat，本节点只写状态+日志、不改路由；
    危机优先级最高，危机与闲聊同现时强制判危机（is_chitchat 压为 False）。
    Step 2 增量：作为每轮首节点统一重置 retrieved_memories，防上轮检索观察经 checkpointer 残留。
    Step 4 增量：同步每轮重置 rewrite_count / review_feedback，为 Reviewer 接线（Step 5/10）预清残留。
    """
    user_id = state.get("user_id", "default_user")
    conversation_id = state.get("conversation_id", "")
    text = str(state["messages"][-1].content)
    # 送判文本 = 最近 2 轮历史 + 当前 query 的结构化拼接，仅用于 Jev 双判（关键词兜底仍只扫原文）
    judged_text = _build_jev_text(state["messages"], text)

    try:
        level, is_chitchat = _jev_judge(judged_text, user_id, conversation_id)
        source = "jev"
    except Exception:
        # Jev 不可用才轮到关键词兜底；关键词层只判危机、无闲聊判定能力，闲聊位保守置 False 走旧链路
        logger.exception(
            "Jev 危机判定失败，回落关键词兜底 user_id={} conversation_id={}",
            user_id, conversation_id,
        )
        level, is_chitchat, source = scan_crisis(text), False, "keyword_fallback"

    is_crisis = level != "none"
    if is_crisis:
        # 冲突规则：危机 + 闲聊同现必判危机（危机最高优先），强制压掉闲聊判定
        if is_chitchat:
            logger.warning(
                "危机与闲聊同现，按危机最高优先压掉闲聊判定 user_id={} conversation_id={}",
                user_id, conversation_id,
            )
        logger.warning(
            "危机信号命中 user_id={} conversation_id={} level={} source={} is_chitchat=False",
            user_id, conversation_id, level, source,
        )
        # 命中危机：crisis_card 交由下游 generate_response 危机分支重新写入，此处不动；
        # is_chitchat 显式写 False，防止 checkpointer 保留上一轮的闲聊真值；
        # 危机轮不经检索与复核，retrieved_memories / review_* 统一重置。
        return {
            "is_crisis": True,
            "crisis_level": level,
            "is_chitchat": False,
            "retrieved_memories": [],
            "rewrite_count": 0,
            "review_feedback": "",
            "support_mode": "",
            "support_mode_confidence": 0.0,
            "support_mode_reason": "",
            "case_formulation": "",
        }

    # 非危机轮：checkpointer 会保留上一轮持久化的 crisis_card，而正常生成分支不再写该字段，
    # 不在这里清空就会把上一次的热线卡误返回给前端。危机信号、闲聊位与检索记忆统一由本节点每轮刷新。
    logger.info(
        "双判完成·非危机 user_id={} conversation_id={} level={} is_chitchat={} source={}",
        user_id, conversation_id, level, is_chitchat, source,
    )
    # retrieved_memories 每轮重置：本轮检索命中时由 search_memory 工具经 Command 写回，
    # 不在此清空就会把上一轮的检索观察残留给本轮 Reviewer（Step 2 纯生产端）。
    # rewrite_count / review_feedback 同步重置：Reviewer 回炉计数与意见只在当轮有效。
    update = {
        "is_crisis": False,
        "crisis_level": level,
        "is_chitchat": is_chitchat,
        "retrieved_memories": [],
        "rewrite_count": 0,
        "review_feedback": "",
        "support_mode": "",
        "support_mode_confidence": 0.0,
        "support_mode_reason": "",
        "case_formulation": "",
    }
    if state.get("crisis_card") is not None:
        logger.info(
            "本轮非危机，清空上轮遗留热线卡 user_id={} conversation_id={}",
            user_id, conversation_id,
        )
        update["crisis_card"] = None
    if state.get("retrieved_memories"):
        logger.info(
            "本轮重置上轮遗留检索记忆 user_id={} conversation_id={}",
            user_id, conversation_id,
        )
    return update


# Jev 送判历史窗口：当前 query 之前最近 N 个用户轮（含其间的助手正文回复）
_JEV_HISTORY_TURNS = 2


def _build_jev_text(messages: list, current: str) -> str:
    """拼接 Jev 送判文本：最近 _JEV_HISTORY_TURNS 轮历史原话 + 当前 query，带段落声明供模型区分。

    crisis_router 是每轮首节点，messages[-1] 即当前 query，历史只在其余消息里取：
    从「倒数第 N 条历史用户消息」起切到当前 query 前；历史不足 N 轮时有多少带多少。
    只保留 human/ai 的正文消息，剔除工具轮（ToolMessage 与带 tool_calls 的 AI 消息，
    含 checkpointer 跨轮残留的 CF 检索中间过程）——那些不是会话原话，混进送判只会干扰判定。
    只做原话拼接、不做任何改写：危机语气词一字不动，指代由模型借历史自行补全。
    纯字符串拼接，零额外 LLM 调用、零延迟。
    """
    past_human = [i for i, m in enumerate(messages[:-1]) if isinstance(m, HumanMessage)]
    history = (
        messages[past_human[max(0, len(past_human) - _JEV_HISTORY_TURNS)] : -1]
        if past_human else []
    )
    lines = [
        f"[{'user' if isinstance(m, HumanMessage) else 'assistant'}] {m.content}"
        for m in history
        if isinstance(m, (HumanMessage, AIMessage))
        and not getattr(m, "tool_calls", None)
    ]
    block = (
        "【历史聊天记录】（仅作上下文参考，用于补全当前句的指代与省略，不要对其判定）\n"
        + "\n".join(lines) + "\n\n"
        if lines else ""
    )
    return block + "【当前用户query】（判定对象）\n" + current


def _jev_judge(text: str, user_id: str, conversation_id: str):
    """向 Jev 同一次 invoke 拿回「危机等级 + 是否闲聊」双判。

    text 为 _build_jev_text 的拼接文本（历史 + 当前 query）。
    返回 (level, is_chitchat)：level 为 "none" / "risk" / "imminent"，is_chitchat 为 bool。
    本函数不捕异常：调用失败直接向上抛，由 crisis_router_node 决定兜底策略。
    """
    answers = jev_classifier.invoke(
        {
            "state": text,
            "questions": {
                "crisis_level": CRISIS_LEVEL_QUESTION,
                "is_chitchat": CHITCHAT_QUESTION,
            },
        }
    ).choices

    # —— 危机等级判定（保持与升级前完全一致的行为）——
    crisis_answer = answers["crisis_level"]
    level = crisis_answer.choice if crisis_answer.choice in _LEVEL_RANK else "none"
    confidence = crisis_answer.confidence
    logger.info(
        "Jev 危机判定完成 user_id={} conversation_id={} level={} confidence={} judged_text={!r}",
        user_id, conversation_id, level, confidence, preview(text),
    )
    # 置信度不足则降级最警戒话术（等级仍保留，不会因此判成无危机）
    if level == "imminent" and confidence < _IMMINENT_MIN_CONFIDENCE:
        logger.warning(
            "Jev 判定 imminent 但置信度不足，降级为 risk user_id={} conversation_id={} confidence={} threshold={}",
            user_id, conversation_id, confidence, _IMMINENT_MIN_CONFIDENCE,
        )
        level = "risk"

    # —— 闲聊判定（Step 1 新增，仅写状态+日志）——
    chitchat_answer = answers["is_chitchat"]
    is_chitchat = chitchat_answer.choice == "yes"
    logger.info(
        "Jev 闲聊判定完成 user_id={} conversation_id={} is_chitchat={} confidence={}",
        user_id, conversation_id, is_chitchat, chitchat_answer.confidence,
    )

    return level, is_chitchat
