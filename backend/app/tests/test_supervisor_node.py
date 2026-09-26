"""Step 7/9 Supervisor 纯路由单测：按 CF 置信度决定放行/回落，高置信再按方向召专家，不调 LLM。"""
import agent.graph.supervisor_node as sv


def test_supervisor_node_makes_no_state_change():
    # 纯路由节点不产生任何状态更新
    out = sv.supervisor_node(
        {"user_id": "u1", "support_mode": "listen", "support_mode_confidence": 0.9}
    )
    assert out == {}


def test_router_high_confidence_dispatches_by_mode():
    assert sv.after_supervisor_router({"support_mode_confidence": 0.9, "support_mode": "listen"}) == "empathic_companion"
    assert sv.after_supervisor_router({"support_mode_confidence": 0.7, "support_mode": "clarify"}) == "socratic_clarifier"
    assert sv.after_supervisor_router({"support_mode_confidence": 0.95, "support_mode": "advise"}) == "gentle_advisor"


def test_router_at_threshold_dispatches_by_mode():
    assert sv.after_supervisor_router({"support_mode_confidence": sv._HIGH_CONFIDENCE, "support_mode": "advise"}) == "gentle_advisor"


def test_router_high_confidence_missing_mode_defaults_empathic():
    # 高置信但方向缺失（理论不该发生，CF 高置信必带合法方向）→ 兜底共情陪伴
    assert sv.after_supervisor_router({"support_mode_confidence": 0.9, "support_mode": ""}) == "empathic_companion"


def test_router_low_confidence_falls_back_steer():
    assert sv.after_supervisor_router({"support_mode_confidence": 0.3, "support_mode": "advise"}) == "steer_direction"


def test_router_missing_confidence_falls_back_steer():
    # 缺字段/CF 解析失败保持低值 → 安全回落引导门
    assert sv.after_supervisor_router({}) == "steer_direction"
    assert sv.after_supervisor_router({"support_mode_confidence": 0.0, "support_mode": "listen"}) == "steer_direction"
