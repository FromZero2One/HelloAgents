"""SessionPlugin - 会话持久化插件

职责：会话保存/加载、环境一致性检查、自动保存
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import hashlib
from ..core.session_store import SessionStore
from .plugins import AgentPlugin, PluginContext


class SessionPlugin(AgentPlugin):
    """会话持久化插件"""
    
    name = "session"
    priority = 60
    
    def __init__(self, config=None):
        super().__init__(config)
        self._session_store: Optional[SessionStore] = None
        self._auto_save_counter = 0
    
    def _initialize(self) -> None:
        if not self.config.session_enabled:
            return
        
        self._session_store = SessionStore(session_dir=self.config.session_dir)
        self.context.session_store = self._session_store
        
        # 初始化会话元数据
        self.context.session_metadata = {
            "created_at": datetime.now().isoformat(),
            "total_tokens": 0,
            "total_steps": 0,
            "duration_seconds": 0
        }
        self.context.start_time = datetime.now()
    
    @property
    def session_store(self) -> Optional[SessionStore]:
        return self._session_store
    
    def maybe_auto_save(self) -> None:
        """检查是否需要自动保存"""
        if not self._session_store or not self.config.auto_save_enabled:
            return
        
        history_plugin = self.context.agent._plugin_manager.get_plugin("history")
        if not history_plugin:
            return
        
        history_len = len(history_plugin.get_history())
        if history_len % self.config.auto_save_interval == 0:
            self._auto_save()
    
    def _auto_save(self) -> None:
        """自动保存（静默失败）"""
        try:
            self.save_session("session-auto")
        except Exception as e:
            if self.config.debug:
                print(f"⚠️ 自动保存失败: {e}")
    
    def save_session(self, session_name: str) -> str:
        """手动保存会话"""
        if not self._session_store:
            raise RuntimeError("会话持久化未启用")
        
        # 更新元数据
        self.context.session_metadata["duration_seconds"] = (
            datetime.now() - self.context.start_time
        ).total_seconds()
        
        # 获取历史
        history_plugin = self.context.agent._plugin_manager.get_plugin("history")
        history = history_plugin.get_history() if history_plugin else []
        
        # 计算工具 Schema 哈希
        tool_schema_hash = self._compute_tool_schema_hash()
        
        # 获取 Read 工具缓存
        read_cache = self._get_read_cache()
        
        filepath = self._session_store.save(
            agent_config=self._get_agent_config(),
            history=history,
            tool_schema_hash=tool_schema_hash,
            read_cache=read_cache,
            metadata=self.context.session_metadata,
            session_name=session_name
        )
        
        return filepath
    
    def load_session(self, filepath: str, check_consistency: bool = True) -> None:
        """加载会话"""
        if not self._session_store:
            raise RuntimeError("会话持久化未启用")
        
        session_data = self._session_store.load(filepath)
        
        # 环境一致性检查
        if check_consistency:
            config_check = self._session_store.check_config_consistency(
                saved_config=session_data.get("agent_config", {}),
                current_config=self._get_agent_config()
            )
            
            if not config_check["consistent"]:
                print("⚠️ 环境配置不一致：")
                for warning in config_check["warnings"]:
                    print(f"  - {warning}")
            
            tool_check = self._session_store.check_tool_schema_consistency(
                saved_hash=session_data.get("tool_schema_hash", ""),
                current_hash=self._compute_tool_schema_hash()
            )
            
            if tool_check["changed"]:
                print(f"⚠️ 工具定义已变化")
                print(f"  建议：{tool_check['recommendation']}")
        
        # 恢复历史
        from ..core.message import Message
        history_plugin = self.context.agent._plugin_manager.get_plugin("history")
        if history_plugin:
            history_plugin.clear_history()
            for msg_data in session_data.get("history", []):
                history_plugin.history_manager.append(Message.from_dict(msg_data))
        
        # 恢复元数据
        self.context.session_metadata = session_data.get("metadata", {})
        
        # 恢复 Read 工具缓存
        if self.context.tool_registry and session_data.get("read_cache"):
            self.context.tool_registry.read_metadata_cache = session_data["read_cache"]
        
        print(f"✅ 会话已恢复：{session_data.get('session_id', 'unknown')}")
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        if not self._session_store:
            return []
        return self._session_store.list_sessions()
    
    def _get_agent_config(self) -> Dict[str, Any]:
        config = {
            "name": self.context.agent.name,
            "agent_type": self.context.agent.__class__.__name__,
            "llm_provider": getattr(self.context.llm, 'provider', 'unknown'),
            "llm_model": getattr(self.context.llm, 'model_id', getattr(self.context.llm, 'model', 'unknown'))
        }
        if hasattr(self.context.agent, 'max_steps'):
            config["max_steps"] = self.context.agent.max_steps
        return config
    
    def _compute_tool_schema_hash(self) -> str:
        if not self.context.tool_registry:
            return "no-tools"
        
        tools_signature = {}
        for tool_name in sorted(self.context.tool_registry.list_tools()):
            tool = self.context.tool_registry.get_tool(tool_name)
            if tool:
                tools_signature[tool_name] = {
                    "name": tool.name,
                    "description": tool.description[:100] if tool.description else "",
                    "parameters": list(tool.parameters.keys()) if hasattr(tool, 'parameters') and tool.parameters else []
                }
        
        schema_str = json.dumps(tools_signature, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]
    
    def _get_read_cache(self) -> Dict[str, Dict]:
        if self.context.tool_registry and hasattr(self.context.tool_registry, 'read_metadata_cache'):
            return self.context.tool_registry.read_metadata_cache
        return {}