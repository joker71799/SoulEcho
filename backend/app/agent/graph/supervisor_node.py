"""Supervisor 路由节点（Step 7 修正为纯路由，Step 8 确认“低置信才暂停”）：不调任何 LLM。

方向与置信度已由上游 CF 一手产出（support_mode + support_mode_confidence）。
本节点只做一个风险门控，决定"要不要打扰用户确认方向"：
- 置信度 >= 阈值 → 放行，进 healing_reply 用 CF 定的方向直接生成（用户无感直通）；
- 置信度 <  阈值（含 CF 解析失败被压到 0.0）→ 回落 steer_direction 引导门，让用户点选
  （Step 8 已落地：引导门会递送 CF 推断的方向+理由 + 可覆盖选项，首轮必问已移除）。

阈值 _HIGH_CONFIDENCE 是"多自信才敢替用户决定陪伴方式"的策略旋钮，归属路由侧，故留在此处；
判断方向本身则完全交给 CF。仍保留为独立节点，是为 Step 8 的 HITL 升级预留落点。
"""
from agent.graph.experts_node import route_by_mode
from agent.graph.state import AgentState
from config.logging_config import logger

# 高置信阈值：>= 则信任 CF 定的方向直接放行；< 则回落引导门询问用户。
_HIGH_CONFIDENCE = 0.6


def supervisor_node(state: AgentState):
    """【纯路由节点】不做任何推断、不产生状态更新；门控决策全在 after_supervisor_router。"""
    user_id = state.get("user_id", "default_user")
    confidence = state.get("support_mode_confidence", 0.0) or 0.0
    logger.info(
        "Supervisor 路由 user_id={} conversation_id={} support_mode={} confidence={:.2f} -> {}",
        user_id, state.get("conversation_id", ""), state.get("support_mode", ""), confidence,
        "放行生成" if confidence >= _HIGH_CONFIDENCE else "回落引导门",
    )
    return {}


def after_supervisor_router(state: AgentState):
    """Supervisor 出口路由：只看 CF 填的置信度。

    - 高置信 → 信任 CF 定的方向，按 support_mode 召回对应专家（Step 9）直接生成；
    - 低置信（含解析失败）→ 回落 steer_direction 引导门，递送推断+理由请用户确认。
    """
    confidence = state.get("support_mode_confidence", 0.0) or 0.0
    return route_by_mode(state) if confidence >= _HIGH_CONFIDENCE else "steer_direction"
