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

# 询问语（暂停时展示给用户）
STEER_QUESTION = "先看看这一刻你更需要我怎么陪你？选一个就好，也可以随时告诉我。"

# 各方向对模型语气/策略的约束，生成回复时按用户选择注入系统提示词。
# key 与 DIRECTION_OPTIONS 的 value 一一对应（单一数据源），由 healing_response_node 引用。
DIRECTION_GUIDE = {
    "listen": "用户此刻希望【被倾听】：以接纳、共情、陪伴为主，多反映和确认情绪，"
              "不急于分析或给建议，让对方真切感到被听见、被接住。",
    "clarify": "用户此刻希望【一起理清】：在共情基础上，用温和的复述与提问，"
               "帮助用户把想法和情绪梳理出一点结构，不替对方下结论。",
    "advise": "用户此刻希望【获得建议】：先充分共情接纳，再给出少量、可操作、贴合处境的小建议，"
              "避免说教和长篇大论。",
}

_VALID_MODES = {o["value"] for o in DIRECTION_OPTIONS}


def steer_direction_node(state: AgentState):
    """
    【节点：疗愈方向引导（Human-in-the-Loop 断点）】
    在生成疗愈回复前，调用 interrupt() 让图「原地暂停」，把“你想我怎样陪你”的
    一键选项递送给前端；用户点选后经 Command(resume=) 恢复，所选方向写入 state
    供 generate_response 定调。人工输入成本极低（点一下），却能为长回复精准定调。
    """
    choice = interrupt(
        {
            "type": "choose_support_mode",
            "question": STEER_QUESTION,
            "options": DIRECTION_OPTIONS,
        }
    )

    # 兼容前端直接回传字符串或 {value:...} 两种形态
    if isinstance(choice, dict):
        choice = choice.get("value")
    # 非法/缺失值统一兜底为“倾听”，保证回复不中断
    support_mode = choice if choice in _VALID_MODES else "listen"

    logger.info(
        "用户选择陪伴方向 user_id={} support_mode={}",
        state.get("user_id", "default_user"), support_mode,
    )
    return {"support_mode": support_mode}
