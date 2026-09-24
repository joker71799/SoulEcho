from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # messages 存储对话历史，Annotated + add_messages 告诉 LangGraph：
    # 当有新消息进来时，不是覆盖旧数据，而是追加（Append）到列表末尾。
    messages: Annotated[list, add_messages]

    # 用户的唯一标识符，用来隔离和检索不同用户的疗愈记忆
    user_id: str

    # 最终返回给前端的疗愈回复文本，全图只由 finalize_reply 节点写入（危机话术/ReAct 产物共用）
    reply: str

    # ===== 危机干预（自伤/自杀意念）=====
    # 是否处于危机：由首节点 crisis_router 判定（Jev 主判，仅其异常时回落关键词扫描）。
    # 作为图的条件边路由位——true 时跳过引导门与 ReAct，直达 finalize_reply 危机话术分支。
    is_crisis: bool

    # 危机等级："none" | "risk" | "imminent"。供危机分支选择对应安全话术模板。
    crisis_level: str

    # 热线求助卡片（结构化 dict，供前端渲染）：命中危机时由 finalize_reply 危机分支写入；
    # 非危机轮如果该变量有值则由首节点 crisis_router 统一置 None，避免 checkpointer 保留上一轮旧卡误返回。
    crisis_card: dict

    # ===== Human-in-the-Loop（场景B：方向引导）=====
    # 用户点选的陪伴方向："listen" / "clarify" / "advise"，
    # 由 steer_direction_node 经 interrupt 暂停、Command(resume=) 恢复后写入，
    # 供 ReAct 生成时定调。会话级持久（靠 checkpointer），选定后本会话沿用。
    support_mode: str