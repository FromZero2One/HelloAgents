"""HistoryPlugin - 历史管理插件

职责：消息追加、历史压缩、轮次管理、序列化
"""

from typing import List, Optional
from ..context.history import HistoryManager
from ..core.message import Message
from .plugins import AgentPlugin, PluginContext


class HistoryPlugin(AgentPlugin):
    """历史管理插件 - 核心插件，始终启用"""
    
    name = "history"
    priority = 10
    
    def __init__(self, config=None):
        super().__init__(config)
        self._history_manager: Optional[HistoryManager] = None
    
    def _initialize(self) -> None:
        self._history_manager = HistoryManager(
            min_retain_rounds=self.config.min_retain_rounds,
            compression_threshold=self.config.compression_threshold
        )
        # 注入到上下文
        self.context.history_manager = self._history_manager
    
    @property
    def history_manager(self) -> HistoryManager:
        return self._history_manager
    
    # ===== 向后兼容 API =====
    
    def add_message(self, message: Message) -> None:
        """添加消息（触发自动压缩检查）"""
        self._history_manager.append(message)
        
        # Token 计数由 TokenCounterPlugin 处理（如果启用）
        token_plugin = self._get_token_plugin()
        if token_plugin:
            token_plugin.on_message_added(message)
        
        # 自动压缩检查
        if self._should_compress():
            self._compress_history()
        
        # 自动保存（由 SessionPlugin 处理）
        session_plugin = self._get_session_plugin()
        if session_plugin and self.config.auto_save_enabled:
            session_plugin.maybe_auto_save()
    
    def clear_history(self) -> None:
        self._history_manager.clear()
        token_plugin = self._get_token_plugin()
        if token_plugin:
            token_plugin.reset()
    
    def get_history(self) -> List[Message]:
        return self._history_manager.get_history()
    
    def _should_compress(self) -> bool:
        token_plugin = self._get_token_plugin()
        if not token_plugin:
            return False
        
        threshold = int(self.config.context_window * self.config.compression_threshold)
        return token_plugin.total_tokens > threshold
    
    def _compress_history(self) -> None:
        history = self._history_manager.get_history()
        
        if self.config.enable_smart_compression:
            summary = self._generate_smart_summary(history)
        else:
            summary = self._generate_simple_summary(history)
        
        self._history_manager.compress(summary)
        
        # 重新计算 Token
        token_plugin = self._get_token_plugin()
        if token_plugin:
            token_plugin.recalculate(self._history_manager.get_history())
    
    def _generate_simple_summary(self, history: List[Message]) -> str:
        rounds = self._history_manager.estimate_rounds()
        user_msgs = sum(1 for msg in history if msg.role == "user")
        assistant_msgs = sum(1 for msg in history if msg.role == "assistant")
        
        return f"""此会话包含 {rounds} 轮对话：
- 用户消息：{user_msgs} 条
- 助手消息：{assistant_msgs} 条
- 总消息数：{len(history)} 条

（历史已压缩，保留最近 {self.config.min_retain_rounds} 轮完整对话）"""
    
    def _generate_smart_summary(self, history: List[Message]) -> str:
        # 智能摘要由 Agent 类直接实现，避免循环引用
        # 这里提供简单摘要作为后备
        return self._generate_simple_summary(history)
    
    def _get_token_plugin(self):
        return self.context.agent._plugin_manager.get_plugin("token_counter")
    
    def _get_session_plugin(self):
        return self.context.agent._plugin_manager.get_plugin("session")