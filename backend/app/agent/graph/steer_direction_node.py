from langgraph.types import interrupt

from agent.graph.state import AgentState
from config.logging_config import logger

# ==========================================
# 场景B：疗愈方向引导（Human-in-the-Loop）文案
# ==========================================
# 与“方向引导”特性强相关，就近定义在本节点（特性归属地），prompt.py 只留真正的提示词。
# value 是回传给后端的稳定标识，label 是展示文本，hint 是悬浮/副标题说明。
DIRECTION_OPTIONS = [
    {"value": "listen", "label": "\U0001faf2 只是想被倾听", "hint": "我会安静陪着你，接住你的情绪"},
    {"value": "clarify", "label": "\U0001f9ed 一起理理清", "hint": "陪你把乱乱的思绪慢慢捋出头绪"},
    {"value": "advise", "label": "\U0001f4a1 想要些建议", "hint": "在被理解之后，给你一点可操作的方向"},
]

# 询问语（Step 8 前：CF 未能给出推断方向时的中性兜底问法）
STEER_QUESTION = "先看看这一刻你更需要我怎么陪你？选一个就好，也可以随时告诉我。"

_VALID_MODES = {o["value"] for o in DIRECTION_OPTIONS}
_LABEL_BY_VALUE = {o["value"]: o["label"] for o in DIRECTION_OPTIONS}


def _build_steer_payload(state: AgentState) -> dict:
    """组装引导门 interrupt 载荷（Step 8：递送 CF 推断方向 + 理由 + 三个可覆盖选项）。

    - CF 高置信时根本不会进到这里；能进来即 CF 把握不足，故把它的推断摊开给用户确认/覆盖；
    - question 文本自带推断方向与理由（对未升级的前端也天然可读）；
    - options 每项附 recommended 标记（命中的即 CF 的推断），并回传 inferred_mode/inferred_reason
      结构化字段，供前端做“推荐高亮/一键采纳”等增强（Step 10 联调）。
    CF 解析失败时 support_mode 为空（已被 crisis_router 每轮清空），退化为中性问法、不带推荐理由。
    """
    inferred = state.get("support_mode") or ""
    # 非法/残留方向一律视为“无推断”，保证问法、recommended、inferred_mode 三者口径一致
    if inferred not in _VALID_MODES:
        inferred = ""
    reason = (state.get("support_mode_reason") or "").strip()
    options = [{**opt, "recommended": opt["value"] == inferred} for opt in DIRECTION_OPTIONS]

    if inferred:
        label = _LABEL_BY_VALUE.get(inferred, inferred)
        question = f"我听下来，你此刻可能更想要「{label.strip()}」"
        if reason:
            question += f"——{reason}"
        question += "。是这样吗？也可以换成此刻更舒服的那个："
    else:
        question = STEER_QUESTION

    return {
        "type": "choose_support_mode",
        "question": question,
        "options": options,
        "inferred_mode": inferred or None,
        "inferred_reason": reason or None,
    }


def steer_direction_node(state: AgentState):
    """
    【节点：疗愈方向引导（Human-in-the-Loop 断点）】
    Step 8 起仅在 CF 低置信时到达（高置信已由 Supervisor 直通、不打扰用户）。
    调用 interrupt() 原地暂停，把「CF 推断方向 + 理由 + 三个可覆盖选项」递送给前端；
    用户采纳或改选后，经 Command(resume=) 恢复，所选方向写入 state 供回复定调。
    """
    choice = interrupt(_build_steer_payload(state))

    # 兼容前端直接回传字符串或 {value:...} 两种形态
    if isinstance(choice, dict):
        choice = choice.get("value")
    # 非法/缺失值统一兜底为“倾听”，保证回复不中断
    support_mode = choice if choice in _VALID_MODES else "listen"

    logger.info(
        "用户点选陪伴方向 user_id={} conversation_id={} support_mode={} (CF 推断={} 置信={:.2f})",
        state.get("user_id", "default_user"), state.get("conversation_id", ""), support_mode,
        state.get("support_mode", "") or "-", state.get("support_mode_confidence", 0.0) or 0.0,
    )
    return {"support_mode": support_mode}
