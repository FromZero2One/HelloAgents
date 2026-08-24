"""TracePlugin - 可观测性插件

职责：TraceLogger 初始化、事件记录
"""

from typing import Optional, Dict, Any
from ..observability.trace_logger import TraceLogger
from .plugins import AgentPlugin, PluginContext


class TracePlugin(AgentPlugin):
    """可观测性插件"""
    
    name = "trace"
    priority = 50
    
    def __init__(self, config=None):
        super().__init__(config)
        self._trace_logger: Optional[TraceLogger] = None
    
    def _initialize(self) -> None:
        if not self.config.trace_enabled:
            return
        
        self._trace_logger = TraceLogger(
            output_dir=self.config.trace_dir,
            sanitize=self.config.trace_sanitize,
            html_include_raw_response=self.config.trace_html_include_raw_response
        )
        self.context.trace_logger = self._trace_logger
        
        # 记录会话开始
        self._trace_logger.log_event(
            "session_start",
            {
                "agent_name": self.context.agent.name,
                "agent_type": self.context.agent.__class__.__name__,
                "config": self.config.model_dump() if hasattr(self.config, 'model_dump') else {}
            }
        )
    
    @property
    def trace_logger(self) -> Optional[TraceLogger]:
        return self._trace_logger
    
    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        if self._trace_logger:
            self._trace_logger.log_event(event_type, data)
    
    def on_agent_start(self, input_text: str) -> None:
        self.log_event("agent_start", {"input": input_text})
    
    def on_step(self, step_data: Dict[str, Any]) -> None:
        self.log_event("step", step_data)
    
    def on_finish(self, result: str) -> None:
        self.log_event("agent_finish", {"result": result})
        # 生成 HTML 报告
        if self._trace_logger:
            self._trace_logger.generate_html_report()
    
    def on_error(self, error: Exception) -> None:
        self.log_event("agent_error", {
            "error": str(error),
            "error_type": type(error).__name__
        })
    
    def teardown(self) -> None:
        if self._trace_logger:
            self._trace_logger.generate_html_report()