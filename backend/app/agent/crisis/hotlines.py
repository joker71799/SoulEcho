"""危机热线资源（单一数据源）。

Jev 主判与关键词兜底两条路径最终都汇聚到 build_crisis_card 生成同一份热线卡片，
保证危机输出一致、可审计。

⚠️ 号码上线前必须逐一核实真实可用（危机场景号码错误的代价极高）。
当前为全国统一热线 + 紧急电话；region 参数为后续“按城市返回当地资源”预留。
"""

# 全国性心理危机/自杀干预热线（默认列表）
NATIONAL_HOTLINES = [
    {"name": "希望24热线", "number": "400-161-9995", "desc": "24小时生命危机干预"},
    {"name": "北京心理危机研究与干预中心", "number": "010-82951332", "desc": "24小时"},
    {"name": "心理援助热线", "number": "12320", "desc": "各地卫生热线，可转心理"},
]

# 紧急电话：仅在即刻危险时随卡片给出
EMERGENCY_HOTLINES = [
    {"name": "报警", "number": "110"},
    {"name": "急救", "number": "120"},
]

# 卡片文案按危机等级分化，与 CRISIS_TEMPLATE 的口径保持一致：
# risk 只做支持性引导（不提紧急电话），imminent 才强调“立刻拨打”。
_CARD = {
    "risk": {
        "title": "你并不孤单，这些专业的人随时可以倾听和支持你",
        "message": "不用一个人扛着，下面的热线随时都可以打。",
    },
    "imminent": {
        "title": "你并不孤单，这一刻请让这些专业的人陪着你",
        "message": "如果你此刻正处于危险之中，请立刻拨打下面的紧急电话。",
    },
}


def build_crisis_card(level: str = "risk", region: str | None = None) -> dict:
    """生成危机热线卡片。

    level 取 "risk" / "imminent"：
    - risk：只给心理援助热线，不放紧急电话（避免对无即刻危险的用户过度惊吓）；
    - imminent：额外给出 110/120 紧急电话，与危机话术中“请立刻拨打”相呼应。
    region 命中城市时（TODO）应前置当地资源，当前统一回落全国列表。

    卡片为只读数据，生成就直接序列化给前端，故直接引用模块常量、不做拷贝。
    """
    card = _CARD.get(level) or _CARD["risk"]
    return {
        "level": level,
        "title": card["title"],
        "message": card["message"],
        "hotlines": NATIONAL_HOTLINES,
        "emergency": EMERGENCY_HOTLINES if level == "imminent" else [],
    }
