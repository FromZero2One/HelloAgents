"""TokenCounterPlugin - Token 计数插件

职责：Token 计数、缓存、增量计算
"""

from typing import List
from ..context.token_counter import TokenCounter
from ..core.message import Message
from .plugins import AgentPlugin, PluginContext


class TokenCounterPlugin(AgentPlugin):
    """Token 计数插件"""
    
    name = "token_counter"
    priority = 20
    
    def __init__(self, config=None):
        super().__init__(config)
        self._token_counter: Optional[TokenCounter] = None
        self._total_tokens = 0
    
    def _initialize(self) -> None:
        model = self.context.llm.model if self.context.llm else "gpt-4"
        self._token_counter = TokenCounter(model=model)
        self.context.token_counter = self._token_counter
    
    @property
    def token_counter(self) -> TokenCounter:
        return self._token_counter
    
    @property
    def total_tokens(self) -> int:
        return self._total_tokens
    
    def on_message_added(self, message: Message) -> None:
        """消息添加时增量更新"""
        new_tokens = self._token_counter.count_message(message)
        self._total_tokens += new_tokens
    
    def reset(self) -> None:
        """重置计数"""
        self._total_tokens = 0
        self._token_counter.clear_cache()
    
    def recalculate(self, history: List[Message]) -> None:
        """重新计算所有历史的 Token"""
        self._total_tokens = self._token_counter.count_messages(history)
    
    def count_message(self, message: Message) -> int:
        return self._token_counter.count_message(message)
    
    def count_messages(self, messages: List[Message]) -> int:
        return self._token_counter.count_messages(messages)