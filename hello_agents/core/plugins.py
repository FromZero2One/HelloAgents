"""Agent Plugin System - 组合式架构核心

设计原则：
- 组合优于继承：Agent 核心只负责 LLM 调用和消息循环
- 插件化：每项能力（历史、工具、会话、追踪、技能、子代理等）独立插件
- 向后兼容：现有 Agent 子类零修改即可工作
- 配置驱动：通过 Config 控制插件启用/禁用
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .agent import Agent
    from .config import Config
    from .llm import HelloAgentsLLM
    from ..tools.registry import ToolRegistry
    from ..observability.trace_logger import TraceLogger
    from ..skills.loader import SkillLoader
    from ..context.history import HistoryManager
    from ..context.token_counter import TokenCounter
    from ..context.truncator import ObservationTruncator
    from ..core.session_store import SessionStore
    from ..tools.tool_filter import ToolFilter
    from ..tools.builtin.task_tool import TaskTool
    from ..tools.builtin.todowrite_tool import TodoWriteTool
    from ..tools.builtin.devlog_tool import DevLogTool


@dataclass
class PluginContext:
    """插件运行时上下文 - 传递给插件的共享状态"""
    agent: 'Agent'
    config: 'Config'
    llm: 'HelloAgentsLLM'
    tool_registry: Optional['ToolRegistry'] = None
    
    # 共享组件（由插件初始化并注册）
    history_manager: Optional['HistoryManager'] = None
    token_counter: Optional['TokenCounter'] = None
    truncator: Optional['ObservationTruncator'] = None
    trace_logger: Optional['TraceLogger'] = None
    skill_loader: Optional['SkillLoader'] = None
    session_store: Optional['SessionStore'] = None
    
    # 会话元数据
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: Any = None  # datetime
    
    def __post_init__(self):
        from datetime import datetime
        if self.start_time is None:
            self.start_time = datetime.now()


class AgentPlugin(ABC):
    """Agent 插件基类
    
    每个插件负责单一职责：
    - 初始化：setup(context) - 注入依赖、注册组件
    - 生命周期：on_agent_start/step/finish/error
    - 清理：teardown()
    """
    
    name: str = "base"
    enabled: bool = True
    priority: int = 0  # 加载优先级，数字越小越先加载
    
    def __init__(self, config: Optional['Config'] = None):
        self.config = config
        self._context: Optional[PluginContext] = None
    
    @property
    def context(self) -> PluginContext:
        if self._context is None:
            raise RuntimeError(f"Plugin {self.name} not initialized. Call setup() first.")
        return self._context
    
    def setup(self, context: PluginContext) -> None:
        """插件初始化 - 注入上下文、创建组件、注册工具等
        
        Args:
            context: 共享运行时上下文
        """
        self._context = context
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """子类实现：创建组件、注册工具、绑定钩子等"""
        pass
    
    def on_agent_start(self, input_text: str) -> None:
        """Agent 开始执行"""
        pass
    
    def on_step(self, step_data: Dict[str, Any]) -> None:
        """每个推理步骤"""
        pass
    
    def on_finish(self, result: str) -> None:
        """Agent 执行完成"""
        pass
    
    def on_error(self, error: Exception) -> None:
        """Agent 执行出错"""
        pass
    
    def teardown(self) -> None:
        """清理资源"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(enabled={self.enabled})"


class PluginManager:
    """插件管理器 - 负责插件加载、排序、生命周期管理"""
    
    def __init__(self):
        self._plugins: List[AgentPlugin] = []
        self._initialized = False
        self._context: Optional[PluginContext] = None
    
    @property
    def context(self) -> Optional[PluginContext]:
        return self._context
    
    def register(self, plugin: AgentPlugin) -> 'PluginManager':
        """注册插件（按 priority 排序）"""
        if plugin.enabled:
            self._plugins.append(plugin)
            self._plugins.sort(key=lambda p: p.priority)
        return self
    
    def register_many(self, plugins: List[AgentPlugin]) -> 'PluginManager':
        for p in plugins:
            self.register(p)
        return self
    
    def initialize_all(self, context: PluginContext) -> None:
        """初始化所有插件"""
        if self._initialized:
            return
        
        self._context = context
        
        for plugin in self._plugins:
            try:
                plugin.setup(context)
            except Exception as e:
                if context.config.debug:
                    print(f"⚠️ Plugin {plugin.name} init failed: {e}")
        
        self._initialized = True
    
    def emit_start(self, input_text: str) -> None:
        for plugin in self._plugins:
            try:
                plugin.on_agent_start(input_text)
            except Exception:
                pass  # 钩子异常不中断
    
    def emit_step(self, step_data: Dict[str, Any]) -> None:
        for plugin in self._plugins:
            try:
                plugin.on_step(step_data)
            except Exception:
                pass
    
    def emit_finish(self, result: str) -> None:
        for plugin in self._plugins:
            try:
                plugin.on_finish(result)
            except Exception:
                pass
    
    def emit_error(self, error: Exception) -> None:
        for plugin in self._plugins:
            try:
                plugin.on_error(error)
            except Exception:
                pass
    
    def teardown_all(self) -> None:
        for plugin in self._plugins:
            try:
                plugin.teardown()
            except Exception:
                pass
    
    def get_plugin(self, name: str) -> Optional[AgentPlugin]:
        for plugin in self._plugins:
            if plugin.name == name:
                return plugin
        return None
    
    def __iter__(self):
        return iter(self._plugins)
    
    def __len__(self):
        return len(self._plugins)


def create_default_plugins(config: 'Config', tool_registry: Optional['ToolRegistry'] = None) -> List[AgentPlugin]:
    """创建默认插件列表（按优先级顺序）"""
    from .plugins_history import HistoryPlugin
    from .plugins_token import TokenCounterPlugin
    from .plugins_tool import ToolPlugin
    from .plugins_truncator import TruncatorPlugin
    from .plugins_trace import TracePlugin
    from .plugins_session import SessionPlugin
    from .plugins_skill import SkillPlugin
    from .plugins_subagent import SubAgentPlugin
    from .plugins_todo import TodoPlugin
    from .plugins_devlog import DevLogPlugin
    
    plugins = [
        HistoryPlugin(config),
        TokenCounterPlugin(config),
        ToolPlugin(config, tool_registry),
        TruncatorPlugin(config),
        TracePlugin(config),
        SessionPlugin(config),
        SkillPlugin(config),
        SubAgentPlugin(config),
        TodoPlugin(config),
        DevLogPlugin(config),
    ]
    
    # 根据配置过滤
    enabled_plugins = []
    for plugin in plugins:
        if _is_plugin_enabled(plugin.name, config, tool_registry):
            enabled_plugins.append(plugin)
        elif config.debug:
            print(f"🔌 Plugin {plugin.name} disabled by config")
    
    return enabled_plugins


def _is_plugin_enabled(plugin_name: str, config: 'Config', tool_registry: Optional['ToolRegistry'] = None) -> bool:
    """根据插件名称和配置判断是否启用"""
    mapping = {
        "history": True,
        "token_counter": True,
        "tool": tool_registry is not None,
        "truncator": True,
        "trace": config.trace_enabled,
        "session": config.session_enabled,
        "skill": config.skills_enabled,
        "subagent": config.subagent_enabled,
        "todo": config.todowrite_enabled,
        "devlog": config.devlog_enabled,
    }
    return mapping.get(plugin_name, True)