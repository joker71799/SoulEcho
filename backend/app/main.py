import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
    """前端发过来的日记请求体数据模型"""
    user_id: str  # 用户的唯一 ID，例如 "user_001"
    content: str  # 用户倾诉的文本内容 / 日记内容


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
    前端通过 POST 请求把用户 ID 和日记内容发过来，经过 Agent 状态图处理后返回疗愈话语。
    """
    try:
        # 1. 组装输入数据，格式必须符合我们在 graph.py 中定义的 AgentState 结构
        inputs = {
            "messages": [payload.content],  # 放入列表作为消息队列的最新一条
            "user_id": payload.user_id  # 用户 ID
        }

        # 2. 触发并运行 LangGraph 状态图工作流
        # invoke 会顺着 START -> 检索记忆 -> 生成回复 -> END 自动跑完一轮
        result = soulecho_agent.invoke(inputs)

        # 3. 从最终返回的 State 结果中取出 reply 字段，封装并返回给前端
        return JournalResponse(reply=result["reply"])

    except Exception as e:
        # 如果中间发生任何未知异常（如向量数据库连接失败、计算报错），向前端抛出 500 错误
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
    uvicorn.run("main:app", host="127.0.0.1", port=9000, reload=True)
