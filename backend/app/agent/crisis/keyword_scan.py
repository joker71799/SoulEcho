"""确定性关键词扫描：Jev 不可用时的危机识别兜底。

当前策略下（见 crisis_router_node）：Jev 为主判，本模块仅在其调用异常时接管判定，
不再作为叠加的“地板”去纠正模型结论。保留纯函数、零成本、不依赖网络，
是为了在 Jev 挂掉时危机识别不至于完全失效。
"""

import re

# 紧迫危险：出现明确自杀/终结生命的表达，命中即定为 imminent
_IMMINENT_PATTERNS = [
    r"想死",
    r"不想活",
    r"自杀",
    r"了(结|断)自己",
    r"一了百了",
    r"结束(自己的)?(生命|性命|活着)",
    r"活不下去",
]

# 风险倾向：自伤、隐晦的轻生/消失念头，命中定为 risk
_RISK_PATTERNS = [
    r"自伤",
    r"伤害自己",
    r"割腕",
    r"不想(再)?醒来",
    r"(消失|不在)了(就好|算了|最好)",
    r"(走不到|撑不到)明天",
]


def scan_crisis(text: str) -> str:
    """返回危机等级："imminent" / "risk" / "none"。纯确定性，不依赖模型。"""
    if not text:
        return "none"
    for pattern in _IMMINENT_PATTERNS:
        if re.search(pattern, text):
            return "imminent"
    for pattern in _RISK_PATTERNS:
        if re.search(pattern, text):
            return "risk"
    return "none"
