"""DevLogPlugin - 开发日志插件

职责：DevLogTool 注册
"""

from typing import Optional
from ..tools.builtin.devlog_tool import DevLogTool
from .plugins import AgentPlugin, PluginContext


class DevLogPlugin(AgentPlugin):
    """开发日志插件"""
    
    name = "devlog"
    priority = 100
    
    def __init__(self, config=None):
        super().__init__(config)
        self._devlog_tool: Optional[DevLogTool] = None
    
    def _initialize(self) -> None:
        if not self.config.devlog_enabled or not self.context.tool_registry:
            return
        
        trace_plugin = self.context.agent._plugin_manager.get_plugin("trace")
        session_id = trace_plugin.trace_logger.session_id if trace_plugin and trace_plugin.trace_logger else self._generate_session_id()
        
        project_root = getattr(self.context.agent, 'working_dir', ".")
        if hasattr(project_root, '__fspath__'):
            project_root = str(project_root)
        
        self._devlog_tool = DevLogTool(
            session_id=session_id,
            agent_name=self.context.agent.name,
            project_root=project_root,
            persistence_dir=self.config.devlog_persistence_dir
        )
        
        self.context.tool_registry.register_tool(self._devlog_tool)
    
    def _generate_session_id(self) -> str:
        import uuid
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        random_suffix = uuid.uuid4().hex[:4]
        return f"s-{timestamp}-{random_suffix}"
    
    @property
    def devlog_tool(self) -> Optional[DevLogTool]:
        return self._devlog_tool