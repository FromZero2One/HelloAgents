"""TruncatorPlugin - 观察截断插件

职责：工具输出截断、完整输出保存
"""

from typing import Optional
from ..context.truncator import ObservationTruncator
from .plugins import AgentPlugin, PluginContext


class TruncatorPlugin(AgentPlugin):
    """观察截断插件"""
    
    name = "truncator"
    priority = 40
    
    def __init__(self, config=None):
        super().__init__(config)
        self._truncator: Optional[ObservationTruncator] = None
    
    def _initialize(self) -> None:
        self._truncator = ObservationTruncator(
            max_lines=self.config.tool_output_max_lines,
            max_bytes=self.config.tool_output_max_bytes,
            truncate_direction=self.config.tool_output_truncate_direction,
            output_dir=self.config.tool_output_dir
        )
        self.context.truncator = self._truncator
    
    @property
    def truncator(self) -> ObservationTruncator:
        return self._truncator
    
    def truncate(self, text: str) -> str:
        """截断文本"""
        if not self._truncator:
            return text
        return self._truncator.truncate(text)
    
    def save_full_output(self, tool_name: str, content: str) -> Optional[str]:
        """保存完整输出到文件"""
        if not self._truncator:
            return None
        return self._truncator.save_full_output(tool_name, content)