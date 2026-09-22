# 从当前包的 graph 模块中，导入编译好的状态机实例
from .graph.graph import soulecho_agent

# 显式定义对外暴露的接口列表
# 当别的地方执行 `from agent import *` 时，只会导入 soulecho_agent
__all__ = ["soulecho_agent"]
