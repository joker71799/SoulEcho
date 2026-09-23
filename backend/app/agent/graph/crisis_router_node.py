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
CRISIS_LEVEL_QUESTION = Choice(
    instructions="用户当前这句话属于哪种自伤/自杀危机等级？",
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

# Jev 会返回校准过的置信度，官方建议「风险容忍度由调用方代码显式声明」：
# 低置信度时不把等级升级到最警戒的 imminent 话术，避免过度惊吓用户；
# 但仍按 risk 出卡，保证宁可多报也不漏报。
_IMMINENT_MIN_CONFIDENCE = 0.6

_LEVEL_RANK = {"none": 0, "risk": 1, "imminent": 2}


def crisis_router_node(state: AgentState):
    """
    【首节点：危机识别】
    以 Jev 的结构化判定为唯一决策来源；仅当 Jev 调用异常（网络/额度/超时等）时，
    才回落确定性关键词扫描，保证 Jev 不可用时危机识别不至于完全失效。
    命中危机的轮次由下游路由直接跳到唯一生成节点走危机分支，跳过检索与方向引导。
    """
    user_id = state.get("user_id", "default_user")
    text = state["messages"][-1].content

    try:
        level, source = _jev_level(text, user_id), "jev"
    except Exception:
        # Jev 不可用才轮到关键词兜底
        logger.exception("Jev 危机判定失败，回落关键词兜底 user_id={}", user_id)
        level, source = scan_crisis(text), "keyword_fallback"

    is_crisis = level != "none"
    if is_crisis:
        logger.warning(
            "危机信号命中 user_id={} level={} source={}", user_id, level, source
        )
        # 命中危机：crisis_card 交由下游 generate_response 危机分支重新写入，此处不动
        return {"is_crisis": True, "crisis_level": level}

    # 非危机轮：checkpointer 会保留上一轮持久化的 crisis_card，而正常生成分支不再写该字段，
    # 不在这里清空就会把上一次的热线卡误返回给前端。危机信号三个字段统一由本节点刷新。
    update = {"is_crisis": False, "crisis_level": level}
    if state.get("crisis_card") is not None:
        logger.info("本轮非危机，清空上轮遗留热线卡 user_id={}", user_id)
        update["crisis_card"] = None
    return update


def _jev_level(text: str, user_id: str) -> str:
    """向 Jev 问一次危机等级，返回 "none" / "risk" / "imminent"。

    本函数不捕异常：调用失败直接向上抛，由 crisis_router_node 决定兜底策略。
    """
    answer = jev_classifier.invoke(
        {"state": text, "questions": {"crisis_level": CRISIS_LEVEL_QUESTION}}
    ).choices["crisis_level"]

    level = answer.choice if answer.choice in _LEVEL_RANK else "none"
    confidence = answer.confidence
    logger.info(
        "Jev 危机判定完成 user_id={} level={} confidence={} query={!r}",
        user_id, level, confidence, text,
    )

    # 置信度不足则降级最警戒话术（等级仍保留，不会因此判成无危机）
    if level == "imminent" and confidence < _IMMINENT_MIN_CONFIDENCE:
        logger.warning(
            "Jev 判定 imminent 但置信度不足，降级为 risk user_id={} confidence={} threshold={}",
            user_id, confidence, _IMMINENT_MIN_CONFIDENCE,
        )
        return "risk"
    return level
