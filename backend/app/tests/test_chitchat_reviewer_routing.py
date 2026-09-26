"""Step 5 接线路由单测：闲聊 G1（命中必核 / 寒暄直出）与 Reviewer 回炉分派。

纯路由函数，不触网、不跑图。
"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph.chitchat_agent_node import after_chitchat_router
from agent.graph.reviewer_node import after_reviewer_router
from agent.graph.graph import after_tools_router

_TOOL_CALL = {"name": "search_memory", "args": {"query": "x"}, "id": "call_1"}


def _messages(*msgs):
    return {"messages": list(msgs)}


def test_chitchat_first_pass_with_toolcall_goes_tools():
    state = _messages(
        HumanMessage(content="你还记得我喜欢什么茶吗"),
        AIMessage(content="", tool_calls=[_TOOL_CALL]),
    )
    assert after_chitchat_router(state) == "tools"


def test_chitchat_hit_routes_to_reviewer():
    # 已出终稿文本、且本轮检索命中 → 命中必核
    state = _messages(
        HumanMessage(content="你还记得我喜欢什么茶吗"),
        AIMessage(content="当然，你喜欢桂花茶"),
    )
    state["retrieved_memories"] = ["用户喜欢桂花茶"]
    assert after_chitchat_router(state) == "reviewer"


def test_chitchat_pure_greeting_goes_finalize():
    # 纯寒暄、无检索命中 → 直出收尾
    state = _messages(
        HumanMessage(content="嗨，在吗"),
        AIMessage(content="在的呀，今天想聊点什么？"),
    )
    state["retrieved_memories"] = []
    assert after_chitchat_router(state) == "finalize_reply"


def test_reviewer_pass_routes_to_finalize():
    assert after_reviewer_router({"review_feedback": ""}) == "finalize_reply"


def test_reviewer_rewrite_routes_back_to_chitchat():
    # 闲聊链回炉：is_chitchat=True → 交回 chitchat_agent（复用已有检索观察不重检索）
    state = {"review_feedback": "别把没有的说成有，坦诚一点", "is_chitchat": True}
    assert after_reviewer_router(state) == "chitchat_agent"


def test_reviewer_rewrite_healing_routes_back_to_origin_expert():
    # 疗愈链回炉（Step 10 谁的稿谁改）：按本轮 support_mode 回到同一位专家
    assert after_reviewer_router(
        {"review_feedback": "太说教", "is_chitchat": False, "support_mode": "advise"}
    ) == "gentle_advisor"
    assert after_reviewer_router(
        {"review_feedback": "别急建议", "is_chitchat": False, "support_mode": "listen"}
    ) == "empathic_companion"


def test_tools_chitchat_branch_returns_to_agent():
    # 合并后共享 tools 节点：闲聊链（is_chitchat=True）检索后回 chitchat_agent
    assert after_tools_router({"is_chitchat": True, "messages": []}) == "chitchat_agent"


def test_tools_cf_branch_delegates_to_round_limit():
    # CF 链（is_chitchat=False）：委托 after_cf_tools_router 限步，本轮一轮检索未超限→回 cf_agent
    state = {
        "is_chitchat": False,
        "messages": [
            HumanMessage(content="最近压力好大"),
            AIMessage(content="", tool_calls=[_TOOL_CALL]),
            ToolMessage(content="obs", tool_call_id="call_1"),
        ],
    }
    assert after_tools_router(state) == "cf_agent"
