"""CF 强制收敛节点：cf_agent 反复检索却迟迟不收尾时的逃生门（从 case_formulation_node 拆出）。

after_cf_tools_router 在工具轮达 MAX_CF_TOOL_ROUNDS 时把流程送来这里——用「不绑工具」的模型
（结构上无法再发 tool_call）强制产出与 cf_agent 末轮同格式的结构化结果。
复用主 CF 模块的 cf_window / cf_update，产出契约与 cf_agent 完全一致，只是这一趟的人设指令换成「收敛出结果」。
"""
from langchain_core.messages import SystemMessage

from agent.graph.case_formulation_node import MAX_CF_TOOL_ROUNDS, cf_update, cf_window
from agent.graph.state import AgentState
from agent.prompt import CF_CONVERGE_DIRECTIVE, CF_SYSTEM_PROMPT
from config.llm_config import chat_llm
from config.logging_config import logger

# 收敛趟：不绑工具，从结构上杜绝再发 tool_call，只能基于已有观察产出结构化结果。
# 与 cf_agent 同属分诊判定，刻意低温（0.0）求稳定可复现（与 Reviewer 同约定）。
cf_converge_llm = chat_llm.bind(temperature=0.0)


def cf_converge_node(state: AgentState):
    """【CF 强制收敛节点】达工具轮上限后用不绑工具模型强制产出结构化结果（而非疗愈回复）。"""
    user_id = state.get("user_id", "default_user")
    conversation_id = state.get("conversation_id", "")
    ai_message = cf_converge_llm.invoke(
        [SystemMessage(content=CF_SYSTEM_PROMPT + CF_CONVERGE_DIRECTIVE), *cf_window(state)]
    )
    logger.warning(
        "CF 达最大工具轮 {}，强制收敛产出结构化结果 user_id={} conversation_id={}",
        MAX_CF_TOOL_ROUNDS, user_id, conversation_id,
    )
    return cf_update(ai_message)
