import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel
# 统一日志（导入即完成 loguru 初始化，必须放在业务模块之前）
from config.logging_config import logger, preview
# 跨文件夹引入你在 agent 包中编写并导出的状态机
from agent import soulecho_agent

# 初始化 FastAPI 实例
app = FastAPI(
    title="SoulEcho API",
    description="个人心理疗愈 Agent MVP 后端接口服务",
    version="0.1.0"
)


# ==========================================
# 定义请求与响应的数据模型 (Pydantic)
# ==========================================

class ChatInput(BaseModel):
    user_id: str          # 用于 Mem0 长期记忆隔离（跨会话）
    conversation_id: str  # 用于 Checkpointer 单次会话内多轮对话上下文隔离（thread_id）
    content: str  # 用户倾诉的内容


class DirectionInput(BaseModel):
    """用户在方向引导门点选后回传的决策"""
    user_id: str
    conversation_id: str
    support_mode: str     # listen / clarify / advise


class ChatResponse(BaseModel):
    """统一响应：done 直接给回复；direction_required 给待点选的引导选项"""
    status: str                          # "done" | "direction_required"
    reply: Optional[str] = None          # status=done 时有值
    steer: Optional[dict] = None         # status=direction_required 时为 interrupt 递送的引导载荷
    crisis: Optional[dict] = None        # 危机热线卡片（命中危机时随 reply 一起返回）


# ==========================================
# 定义 HTTP 路由定义
# ==========================================

def _thread_config(conversation_id: str):
    return {"configurable": {"thread_id": conversation_id}}


def _to_response(result: dict, user_id: str, conversation_id: str) -> ChatResponse:
    # 图在 steer_direction 处 interrupt -> 返回体带 __interrupt__，转成“待点选方向”响应
    if "__interrupt__" in result and result["__interrupt__"]:
        steer_payload = result["__interrupt__"][0].value
        logger.info(
            "图已暂停等待方向引导 user_id={} conversation_id={} question={!r}",
            user_id, conversation_id, preview(steer_payload.get("question", "")),
        )
        return ChatResponse(status="direction_required", steer=steer_payload)

    # 正常跑完（或用户点选恢复后）：记录 State 快照，返回最终疗愈回复
    # 截断原则：AI/记忆等长正文做预览，用户消息（query）保留全文
    snapshot_msgs = [
        f"{m.type}:{preview(str(m.content)) if m.type == 'ai' else str(m.content)}"
        for m in result.get("messages", [])
    ]
    # crisis_card 是含长文案的结构化 dict，日志只留可审计摘要：等级 + 心理热线数 + 紧急电话数
    card = result.get("crisis_card")
    card_summary = (
        f"level={card.get('level', '')} "
        f"hotlines={len(card.get('hotlines', []))} "
        f"emergency={len(card.get('emergency', []))}"
        if card
        else "-"
    )
    # retrieved_memories 是记忆长文本，逐条 preview 截断，只留可审计的命中概览
    memories = result.get("retrieved_memories") or []
    memory_summaries = " | ".join(preview(str(m)) for m in memories) or "-"
    logger.info(
        "State 快照 user_id={} conversation_id={} is_crisis={} crisis_level={} "
        "crisis_card=[{}] is_chitchat={} support_mode={} confidence={} "
        "support_mode_reason={} case_formulation={} rewrite_count={} review_feedback={} "
        "retrieved_memories({})=[{}] msg_count={} messages=[{}]",
        user_id, conversation_id, result.get("is_crisis", ""),
        result.get("crisis_level", ""), card_summary,
        result.get("is_chitchat", ""),
        result.get("support_mode", ""),
        result.get("support_mode_confidence", 0.0),
        preview(result.get("support_mode_reason", "")),
        preview(result.get("case_formulation", "")),
        result.get("rewrite_count", 0),
        preview(result.get("review_feedback", "")),
        len(memories),
        memory_summaries,
        len(result.get("messages", [])),
        " | ".join(snapshot_msgs),
    )
    return ChatResponse(status="done", reply=result["reply"], crisis=result.get("crisis_card"))


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_with_agent(payload: ChatInput):
    """
    核心对话接口。若会话尚未确定陪伴方向，图会在生成前暂停并返回 direction_required，
    等待前端调用 /api/v1/resume 提交点选后恢复生成。
    """
    try:
        # 用户 query 按原则记全文不做截断
        logger.info(
            "收到对话请求 user_id={} conversation_id={} content={!r}",
            payload.user_id, payload.conversation_id, payload.content,
        )

        # 1. 组装输入数据，格式必须符合我们在 graph.py 中定义的 AgentState 结构
        inputs = {
            "messages": [payload.content],  # 放入列表作为消息队列的最新一条
            "user_id": payload.user_id,  # 用户 ID
            # 会话 ID 随 user_id 一起落进状态，供各节点日志按会话追踪链路
            "conversation_id": payload.conversation_id,
        }

        # 2. 触发并运行 LangGraph 状态图工作流（可能在中途因 interrupt 暂停）
        started = time.perf_counter()
        result = soulecho_agent.invoke(inputs, _thread_config(payload.conversation_id))
        elapsed_ms = (time.perf_counter() - started) * 1000

        # 3. 根据图是否处于中断态，返回“待点选方向”或“最终回复”
        logger.info(
            "对话请求处理完成 user_id={} conversation_id={} elapsed={:.0f}ms",
            payload.user_id, payload.conversation_id, elapsed_ms,
        )
        return _to_response(result, payload.user_id, payload.conversation_id)

    except Exception as e:
        # 记录完整堆栈后再向前端抛 500：detail 只给前端看，堆栈只留在日志里
        logger.exception(
            "Agent 运行异常 user_id={} conversation_id={}",
            payload.user_id, payload.conversation_id,
        )
        raise HTTPException(status_code=500, detail=f"Agent 运行异常: {str(e)}")


@app.post("/api/v1/resume", response_model=ChatResponse)
async def resume_with_direction(payload: DirectionInput):
    """用户在引导门点选方向后，用同一 thread_id 恢复图执行并生成疗愈回复。"""
    try:
        logger.info(
            "收到方向点选 user_id={} conversation_id={} support_mode={}",
            payload.user_id, payload.conversation_id, payload.support_mode,
        )
        started = time.perf_counter()
        # Command(resume=...) 的值会作为 steer_direction_node 里 interrupt() 的返回值注入回来
        result = soulecho_agent.invoke(
            Command(resume=payload.support_mode),
            _thread_config(payload.conversation_id),
        )
        logger.info(
            "恢复执行完成 user_id={} conversation_id={} elapsed={:.0f}ms",
            payload.user_id, payload.conversation_id, (time.perf_counter() - started) * 1000,
        )
        return _to_response(result, payload.user_id, payload.conversation_id)
    except Exception as e:
        logger.exception(
            "恢复执行异常 user_id={} conversation_id={}",
            payload.user_id, payload.conversation_id,
        )
        raise HTTPException(status_code=500, detail=f"恢复执行异常: {str(e)}")


@app.get("/health")
def health_check():
    """健康检查接口，用来确认后端服务是否存活"""
    return {"status": "healthy", "project": "soulecho"}


# ==========================================
# 本地测试启动入口
# ==========================================
if __name__ == "__main__":
    # 当直接运行本文件时，启动 Uvicorn 服务器
    # host='127.0.0.1' 绑定本地，port=9000 端口，reload=True 开启热重载（改动代码自动重启服务器）
    uvicorn.run("main:app", host="127.0.0.1", port=9000, reload=False)
