"""Step 7 修正版 CF 单测：末轮结构化产出（评估+方向+置信度）/ 带检索继续循环 / 超限强制收敛。

不触网：monkeypatch 掉 CF 的两个 LLM。
"""
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent.graph.case_formulation_node as cf
import agent.graph.cf_converge_node as cfc

_TC = {"name": "search_memory", "args": {"query": "x"}, "id": "c1"}


class _FakeLLM:
    def __init__(self, content="", tool_calls=None):
        self._msg = AIMessage(content=content, tool_calls=tool_calls or [])

    def invoke(self, messages):
        return self._msg


def _state(*extra):
    return {"messages": [HumanMessage(content="像上次那样处理吧"), *extra], "user_id": "u1"}


def _cf_json(case="用户当下偏焦虑，需求被安抚", mode="listen", conf=0.85, reason="只是想被听见"):
    return json.dumps(
        {"case_formulation": case, "support_mode": mode, "confidence": conf, "reason": reason},
        ensure_ascii=False,
    )


def test_cf_zero_retrieval_fills_all_three(monkeypatch):
    monkeypatch.setattr(cf, "cf_llm", _FakeLLM(content=_cf_json()))
    out = cf.cf_agent_node(_state())
    assert out["case_formulation"] == "用户当下偏焦虑，需求被安抚"
    assert out["support_mode"] == "listen"
    assert out["support_mode_confidence"] == 0.85
    assert out["support_mode_reason"] == "只是想被听见"


def test_cf_low_confidence_still_writes_mode(monkeypatch):
    # CF 低置信也照常写 support_mode，是否放行由 Supervisor 按置信度决定
    monkeypatch.setattr(cf, "cf_llm", _FakeLLM(content=_cf_json(mode="advise", conf=0.2)))
    out = cf.cf_agent_node(_state())
    assert out["support_mode"] == "advise"
    assert out["support_mode_confidence"] == 0.2


def test_cf_with_toolcall_only_appends_message(monkeypatch):
    monkeypatch.setattr(cf, "cf_llm", _FakeLLM(content="", tool_calls=[_TC]))
    out = cf.cf_agent_node(_state())
    # 检索中间趟只落 AIMessage，不写三态
    assert set(out.keys()) == {"messages"}
    assert len(out["messages"][0].tool_calls) == 1
    assert out["messages"][0].tool_calls[0]["name"] == "search_memory"


def test_cf_unparseable_degrades_to_low_confidence(monkeypatch):
    monkeypatch.setattr(cf, "cf_llm", _FakeLLM(content="模型这次没按 JSON 输出"))
    out = cf.cf_agent_node(_state())
    assert out["case_formulation"] == "模型这次没按 JSON 输出"
    assert out["support_mode"] == ""
    assert out["support_mode_confidence"] == 0.0


def test_cf_invalid_mode_forces_zero_confidence(monkeypatch):
    # 方向非法则置信度失去意义：压到 0.0；support_mode 归空串（不污染 str 字段）
    monkeypatch.setattr(cf, "cf_llm", _FakeLLM(content=_cf_json(mode="nonsense", conf=0.95)))
    out = cf.cf_agent_node(_state())
    assert out["support_mode"] == ""
    assert out["support_mode_confidence"] == 0.0


def test_after_cf_agent_routes_tools_or_supervisor():
    has_tool = _state(AIMessage(content="", tool_calls=[_TC]))
    done = _state(AIMessage(content=_cf_json()))
    assert cf.after_cf_agent_router(has_tool) == "tools"
    assert cf.after_cf_agent_router(done) == "supervisor"


def test_after_cf_tools_limits_rounds():
    one_round = _state(
        AIMessage(content="", tool_calls=[_TC]),
        ToolMessage(content="obs", tool_call_id="c1"),
    )
    two_rounds = _state(
        AIMessage(content="", tool_calls=[_TC]),
        ToolMessage(content="obs", tool_call_id="c1"),
        AIMessage(content="", tool_calls=[_TC]),
        ToolMessage(content="obs", tool_call_id="c1"),
    )
    assert cf.after_cf_tools_router(one_round) == "cf_agent"
    assert cf.after_cf_tools_router(two_rounds) == "cf_converge"


def test_cf_converge_forces_structured_result(monkeypatch):
    monkeypatch.setattr(cfc, "cf_converge_llm", _FakeLLM(content=_cf_json(case="强制收敛出的评估", mode="clarify", conf=0.7)))
    out = cfc.cf_converge_node(_state())
    assert out["case_formulation"] == "强制收敛出的评估"
    assert out["support_mode"] == "clarify"
    assert out["support_mode_confidence"] == 0.7
    assert isinstance(out["messages"][0], AIMessage)
