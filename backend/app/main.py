import time

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# 统一日志（导入即完成 loguru 初始化，必须放在业务模块之前）
from config.logging_config import logger, preview
# 跨文件夹引入你在 agent 包中编写并导出的状态机
from agent import soulecho_agent

# 初始化 FastAPI 实例
app = FastAPI(
    title="SoulEcho API",
    description="个人日志疗愈 Agent MVP 后端接口服务",
    version="0.1.0"
)


# ==========================================
# 定义请求与响应的数据模型 (Pydantic)
# ==========================================

class JournalInput(BaseModel):
    user_id: str          # 用于 Mem0 长期记忆隔离（跨会话）
    conversation_id: str  # 用于 Checkpointer 单次会话内多轮对话上下文隔离（thread_id）
    content: str  # 用户倾诉的日志内容


class JournalResponse(BaseModel):
    """返回给前端的响应体数据模型"""
    reply: str  # Agent 经过疗愈思考后给出的回复内容


# ==========================================
# 定义 HTTP 路由定义
# ==========================================

@app.post("/api/v1/chat", response_model=JournalResponse)
async def chat_with_agent(payload: JournalInput):
    """
    核心对话/日志分析接口。
    前端通过 POST 请求把用户 ID、会话Id、日记内容发过来，经过 Agent 状态图处理后返回疗愈话语。
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
            "user_id": payload.user_id  # 用户 ID
        }

        # 2. 触发并运行 LangGraph 状态图工作流
        # invoke 会顺着 START -> 检索记忆 -> 生成回复 -> END 自动跑完一轮
        # 整体计时：这是发现"慢在哪一环"的最直接手段
        config = {"configurable": {"thread_id": payload.conversation_id}}
        started = time.perf_counter()
        result = soulecho_agent.invoke(inputs, config)
        elapsed_ms = (time.perf_counter() - started) * 1000

        # 3. 记录 graph.invoke 返回的最终 State 快照：本轮的“事实全集”，便于排查记忆召回与多轮消息累积
        # 截断原则：AI/记忆等长正文（memory_context、AI 消息）做预览，用户消息（query）保留全文
        snapshot_msgs = [
            f"{m.type}:{preview(str(m.content)) if m.type == 'ai' else str(m.content)}"
            for m in result.get("messages", [])
        ]
        logger.info(
            "State 快照 user_id={} conversation_id={} msg_count={} memory_preview={!r} messages=[{}]",
            payload.user_id, payload.conversation_id, len(result.get("messages", [])),
            preview(result.get("memory_context", "")), " | ".join(snapshot_msgs),
        )

        # 4. 请求完成,记录日志
        logger.info(
            "对话请求完成 user_id={} conversation_id={} elapsed={:.0f}ms",
            payload.user_id, payload.conversation_id, elapsed_ms
        )
        return JournalResponse(reply=result["reply"])

    except Exception as e:
        # 记录完整堆栈后再向前端抛 500：detail 只给前端看，堆栈只留在日志里
        logger.exception(
            "Agent 运行异常 user_id={} conversation_id={}",
            payload.user_id, payload.conversation_id,
        )
        raise HTTPException(status_code=500, detail=f"Agent 运行异常: {str(e)}")


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
