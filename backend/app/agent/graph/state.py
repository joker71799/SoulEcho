from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

# 只保留最近 N 轮原生对话喂给模型，防止长会话撑爆上下文。
# CF / 闲聊 / 专家等各生成节点共用的消息窗口常量（各节点再按需叠加本轮工具轮余量）。
RECENT_REPLY_TURNS = 5

class AgentState(TypedDict):
    # messages 存储对话历史，Annotated + add_messages 告诉 LangGraph：
    # 当有新消息进来时，不是覆盖旧数据，而是追加（Append）到列表末尾。
    messages: Annotated[list, add_messages]

    # 用户的唯一标识符，用来隔离和检索不同用户的疗愈记忆
    user_id: str

    # 会话唯一标识符（与 checkpointer 的 thread_id 同值）：随 user_id 一起打进各节点日志，
    # 便于多会话并发时按会话追踪完整链路。会话内恒定，故无需每轮重置。
    conversation_id: str

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
    # 本轮的陪伴方向："listen" / "clarify" / "advise"。
    # Step 7 修正后改为「每轮字段」：由 CF Agent 每轮推断填充；CF 低置信时回落
    # steer_direction 经 interrupt 让用户点选覆盖。首节点 crisis_router 每轮清空为 ""，
    # 不再会话级沿用（不再依赖 checkpointer 跨轮保留）。
    support_mode: str

    # ===== 闲聊判定（Step 1 纯增量：仅写状态 + 日志，本轮不参与路由）=====
    # 是否只是闲聊/寒暄：由首节点 crisis_router 与危机等级在同一次 Jev invoke 中「双判」拿回。
    # 冲突规则：危机与闲聊同现必判危机（危机最高优先），此时 is_chitchat 被强制压为 False。
    # 每轮由 crisis_router 统一刷新，防止 checkpointer 保留上一轮旧值；真正的闲聊路由在后续步骤接入。
    is_chitchat: bool

    # ===== 本轮检索命中记忆（Step 2 纯数据铺垫：只有生产端，暂无消费方）=====
    # search_memory 本轮命中且过 MEM0_SEARCH_MIN_SCORE 阈值的记忆条目（文本列表）。
    # 写入端：工具命中时经 Command 写回；清空端：crisis_router 作为每轮首节点统一置 []，
    # 防 checkpointer 把上一轮的检索观察残留给本轮（与 crisis_card 同构的清残留模式）。
    # 后续 Reviewer 忠实度复核（Step 4/5）将以“非空才核”作为触发信号。
    retrieved_memories: list

    # ===== 质量复核（Reviewer，Step 4：本步只建节点与状态位，Step 5/10 接线）=====
    # 本轮已回炉重稿次数：Reviewer 判 rewrite 时 +1；达 MAX_REWRITE_ROUNDS 上限则降级放行（取最后一版）。
    # rewrite_count / review_feedback 均由首节点 crisis_router 每轮重置，防 checkpointer 跨轮残留。
    rewrite_count: int

    # Reviewer 给出的修改意见文本：“空串 = 放行（pass 或降级）”，非空 = 需回炉重写（供后续条件边判定）。
    # 职责边界：Reviewer 只评草稿表面质量（安全基调/忠实度），不做方向判断，故意见只限于措辞/事实层面。
    review_feedback: str

    # ===== Supervisor 路由（Step 6 引入，Step 7 修正为纯路由）=====
    # 本轮对 support_mode 的置信度 [0,1]：由 CF Agent 在分诊时一手填充（不再由 Supervisor 推断）。
    # Supervisor 只按它做纯确定性路由：高置信（>= 阈值）放行、低置信回落引导门询问。
    # 每轮由 crisis_router 重置为 0.0（CF 解析失败会保持低值，从而安全回落引导门）。
    support_mode_confidence: float

    # ===== HITL 低置信递送（Step 8：递送推断方向 + 理由 + 可覆盖选项）=====
    # CF 对本次推断方向的一句话理由（结构化 JSON 的 reason 字段）：
    # 高置信直通时仅写日志；低置信回落引导门时随选项递送给用户参考/覆盖。
    # 每轮由 crisis_router 重置为 ""，防 checkpointer 跨轮残留。
    support_mode_reason: str

    # ===== 个案评估（CF Agent，Step 7：全图唯一 ReAct 循环持有者的产物）=====
    # CF 基于本轮倾诉（必要时多步检索历史）形成的简短个案评估文本，供后续疗愈回复引用。
    # CF 末轮的同一份结构化 JSON 还额外产出 support_mode + confidence（见上）。
    # 同轮 CF 最多完整执行一次（由图拓扑保证：只有 crisis_router 正常分支能进入 CF）；
    # 每轮由 crisis_router 重置为 ""，防 checkpointer 把上一轮评估残留给本轮。
    case_formulation: str