"""Reviewer 节点单测：覆盖 pass / rewrite / 超限降级三路径，外加动态检查项与解析兜底。

不触网：monkeypatch 掉低温复核 LLM，只验节点自身的计数与放行逻辑。
"""
from langchain_core.messages import AIMessage, HumanMessage

import agent.graph.reviewer_node as rv


class _FakeReviewerLLM:
    """替身复核模型：固定返回给定文本，并记录收到的消息以便断言动态检查项。"""

    def __init__(self, content: str):
        self._content = content
        self.seen_messages = None

    def invoke(self, messages):
        self.seen_messages = messages
        return AIMessage(content=self._content)


def _state(**overrides):
    state = {
        "messages": [
            HumanMessage(content="我最近真的很累"),
            AIMessage(content="这段是待审的疗愈草稿"),
        ],
        "user_id": "u1",
        "is_chitchat": False,
        "retrieved_memories": [],
        "rewrite_count": 0,
        "review_feedback": "",
    }
    state.update(overrides)
    return state


def test_pass_keeps_count_and_clears_feedback(monkeypatch):
    monkeypatch.setattr(rv, "reviewer_llm", _FakeReviewerLLM('{"verdict":"pass","feedback":""}'))
    out = rv.reviewer_node(_state(rewrite_count=1))
    assert out == {"rewrite_count": 1, "review_feedback": ""}


def test_rewrite_under_limit_increments_and_writes_feedback(monkeypatch):
    fake = _FakeReviewerLLM('{"verdict":"rewrite","feedback":"太说教了，先共情再给建议"}')
    monkeypatch.setattr(rv, "reviewer_llm", fake)
    out = rv.reviewer_node(_state(rewrite_count=0))
    assert out["rewrite_count"] == 1
    assert out["review_feedback"] == "太说教了，先共情再给建议"


def test_rewrite_over_limit_degrades_to_pass(monkeypatch):
    fake = _FakeReviewerLLM('{"verdict":"rewrite","feedback":"仍然不合格"}')
    monkeypatch.setattr(rv, "reviewer_llm", fake)
    out = rv.reviewer_node(_state(rewrite_count=rv.MAX_REWRITE_ROUNDS))
    # 超限：取最后一版放行——意见清空、计数不再增长
    assert out["review_feedback"] == ""
    assert out["rewrite_count"] == rv.MAX_REWRITE_ROUNDS


def test_last_rewrite_before_limit_still_counts(monkeypatch):
    # rewrite_count = MAX-1 时仍应允许这次回炉，计数升到上限
    fake = _FakeReviewerLLM('{"verdict":"rewrite","feedback":"改一下"}')
    monkeypatch.setattr(rv, "reviewer_llm", fake)
    out = rv.reviewer_node(_state(rewrite_count=rv.MAX_REWRITE_ROUNDS - 1))
    assert out["rewrite_count"] == rv.MAX_REWRITE_ROUNDS
    assert out["review_feedback"] == "改一下"


def test_chitchat_hit_enables_faithfulness_only(monkeypatch):
    fake = _FakeReviewerLLM('{"verdict":"pass","feedback":""}')
    monkeypatch.setattr(rv, "reviewer_llm", fake)
    rv.reviewer_node(_state(is_chitchat=True, retrieved_memories=["用户喜欢桂花茶"]))
    system = fake.seen_messages[0].content
    assert "忠实度" in system
    assert "安全基调" not in system


def test_healing_chain_enables_safety_check(monkeypatch):
    fake = _FakeReviewerLLM('{"verdict":"pass","feedback":""}')
    monkeypatch.setattr(rv, "reviewer_llm", fake)
    rv.reviewer_node(_state(is_chitchat=False, retrieved_memories=[]))
    system = fake.seen_messages[0].content
    assert "安全基调" in system
    assert "忠实度" not in system


def test_unparseable_output_fails_open_to_pass(monkeypatch):
    monkeypatch.setattr(rv, "reviewer_llm", _FakeReviewerLLM("模型这次没按 JSON 输出"))
    out = rv.reviewer_node(_state(rewrite_count=0))
    assert out == {"rewrite_count": 0, "review_feedback": ""}
