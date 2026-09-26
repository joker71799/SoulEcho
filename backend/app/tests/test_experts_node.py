"""Step 9 专家小组单测：按方向分派 + 三专家单发不绑工具、角色各异并引用个案评估。

不触网：monkeypatch 掉 expert_llm，捕获实际拼出的 SystemMessage 以校验角色差异。
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import agent.graph.experts_node as ex


class _RecordingLLM:
    def __init__(self):
        self.system_seen = None

    def invoke(self, messages):
        # 记下拼进去的 system 文案，返回一条纯文本 AIMessage（无 tool_calls）
        self.system_seen = next(m.content for m in messages if isinstance(m, SystemMessage))
        return AIMessage(content="好的，我在听。")


def _state():
    return {
        "messages": [HumanMessage(content="最近好累"), AIMessage(content="辛苦了")],
        "user_id": "u1",
    }


def test_route_by_mode_maps_each_direction():
    assert ex.route_by_mode({"support_mode": "listen"}) == "empathic_companion"
    assert ex.route_by_mode({"support_mode": "clarify"}) == "socratic_clarifier"
    assert ex.route_by_mode({"support_mode": "advise"}) == "gentle_advisor"


def test_route_by_mode_defaults_empathic_on_missing_or_invalid():
    assert ex.route_by_mode({"support_mode": ""}) == "empathic_companion"
    assert ex.route_by_mode({}) == "empathic_companion"
    assert ex.route_by_mode({"support_mode": "nonsense"}) == "empathic_companion"


def test_expert_node_single_shot_appends_plain_message(monkeypatch):
    fake = _RecordingLLM()
    monkeypatch.setattr(ex, "expert_llm", fake)
    out = ex.empathic_companion_node(_state())
    assert list(out.keys()) == ["messages"]
    assert isinstance(out["messages"][0], AIMessage)
    assert not out["messages"][0].tool_calls


def test_three_experts_have_distinct_role_prompts(monkeypatch):
    captured = {}
    for name, node in {
        "empathic_companion": ex.empathic_companion_node,
        "socratic_clarifier": ex.socratic_clarifier_node,
        "gentle_advisor": ex.gentle_advisor_node,
    }.items():
        fake = _RecordingLLM()
        monkeypatch.setattr(ex, "expert_llm", fake)
        node(_state())
        captured[name] = fake.system_seen
    # 三者共享疗愈人设基座，但各自带不同的"陪伴方式"角色段
    assert len({captured["empathic_companion"], captured["socratic_clarifier"], captured["gentle_advisor"]}) == 3
    assert "共情陪伴" in captured["empathic_companion"]
    assert "一起理清" in captured["socratic_clarifier"]
    assert "温和建议" in captured["gentle_advisor"]


def test_expert_injects_case_formulation_block(monkeypatch):
    fake = _RecordingLLM()
    monkeypatch.setattr(ex, "expert_llm", fake)
    state = _state()
    state["case_formulation"] = "用户近期工作压力大，需求被理解"
    ex.socratic_clarifier_node(state)
    assert "个案评估参考" in fake.system_seen
    assert "用户近期工作压力大，需求被理解" in fake.system_seen


def test_expert_applies_rewrite_block(monkeypatch):
    # 首次生成（无 feedback）不带回炉指令；带 feedback 时拼上定向重稿段
    fake = _RecordingLLM()
    monkeypatch.setattr(ex, "expert_llm", fake)
    ex.gentle_advisor_node(_state())
    assert "回炉定向重稿" not in fake.system_seen

    fake2 = _RecordingLLM()
    monkeypatch.setattr(ex, "expert_llm", fake2)
    state = _state()
    state["review_feedback"] = "太长、有点说教，缩到一两点建议"
    ex.gentle_advisor_node(state)
    assert "回炉定向重稿" in fake2.system_seen
    assert "太长、有点说教，缩到一两点建议" in fake2.system_seen
