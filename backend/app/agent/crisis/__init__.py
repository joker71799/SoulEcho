# 危机干预（自伤/自杀意念）能力包。
# 对外暴露热线卡片构造、关键词扫描、安全话术模板三个数据源，
# 供 crisis_router_node（识别）与 healing_response_node（危机分支回复）复用，保证输出一致。
from agent.crisis.hotlines import build_crisis_card
from agent.crisis.keyword_scan import scan_crisis
from agent.crisis.templates import CRISIS_TEMPLATE

__all__ = ["build_crisis_card", "scan_crisis", "CRISIS_TEMPLATE"]
