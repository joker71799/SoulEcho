"""Step 8 HITL 引导载荷单测：低置信时递送 CF 推断方向 + 理由 + 可覆盖选项，及 resume 兼容。

不触发真正的 interrupt：只验纯函数 _build_steer_payload 与方向兜底逻辑。
"""
import agent.graph.steer_direction_node as sd


def test_payload_with_inference_carries_direction_reason_and_recommendation():
    state = {"support_mode": "clarify", "support_mode_reason": "思绪很乱想理一理"}
    payload = sd._build_steer_payload(state)
    # question 文本自带推断方向与理由（未升级前端也能直接读）
    assert "一起理理清" in payload["question"]
    assert "思绪很乱想理一理" in payload["question"]
    # 结构化字段回传，供 Step 10 前端做推荐高亮
    assert payload["inferred_mode"] == "clarify"
    assert payload["inferred_reason"] == "思绪很乱想理一理"
    # 三个可覆盖选项齐全，命中项标 recommended
    assert [o["value"] for o in payload["options"]] == ["listen", "clarify", "advise"]
    rec = {o["value"]: o["recommended"] for o in payload["options"]}
    assert rec == {"listen": False, "clarify": True, "advise": False}


def test_payload_without_inference_falls_back_neutral():
    # CF 解析失败：support_mode 被每轮清空为空 → 中性问法，不带推荐理由，无推荐项
    payload = sd._build_steer_payload({"support_mode": "", "support_mode_reason": ""})
    assert payload["question"] == sd.STEER_QUESTION
    assert payload["inferred_mode"] is None
    assert payload["inferred_reason"] is None
    assert all(o["recommended"] is False for o in payload["options"])


def test_payload_inference_without_reason_still_asks():
    payload = sd._build_steer_payload({"support_mode": "listen", "support_mode_reason": ""})
    assert "只是想被倾听" in payload["question"]
    assert payload["inferred_mode"] == "listen"
    assert payload["inferred_reason"] is None


def test_invalid_inferred_mode_treated_as_no_inference():
    # 万一残留非法方向，等同无推断，走中性问法
    payload = sd._build_steer_payload({"support_mode": "nonsense", "support_mode_reason": "x"})
    assert payload["question"] == sd.STEER_QUESTION
    assert payload["inferred_mode"] is None
