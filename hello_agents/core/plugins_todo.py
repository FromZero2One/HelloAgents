"""TodoPlugin - 进度管理插件

职责：TodoWriteTool 注册
"""

from typing import Optional
from ..tools.builtin.todowrite_tool import TodoWriteTool
from .plugins import AgentPlugin, PluginContext


class TodoPlugin(AgentPlugin):
    """进度管理插件"""
    
    name = "todo"
    priority = 90
    
    def __init__(self, config=None):
        super().__init__(config)
        self._todo_tool: Optional[TodoWriteTool] = None
    
    def _initialize(self) -> None:
        if not self.config.todowrite_enabled or not self.context.tool_registry:
            return
        
        project_root = getattr(self.context.agent, 'working_dir', ".")
        if hasattr(project_root, '__fspath__'):
            project_root = str(project_root)
        
        self._todo_tool = TodoWriteTool(
            project_root=project_root,
            persistence_dir=self.config.todowrite_persistence_dir
        )
        
        self.context.tool_registry.register_tool(self._todo_tool)
    
    @property
    def todo_tool(self) -> Optional[TodoWriteTool]:
        return self._todo_tool