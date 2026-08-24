"""SubAgentPlugin - 子代理机制插件

职责：TaskTool 注册、子代理执行（上下文隔离）、工具过滤
"""

from typing import Optional, Dict, Any, List
from ..agents.factory import default_subagent_factory
from ..tools.builtin.task_tool import TaskTool
from ..tools.tool_filter import ToolFilter
from .plugins import AgentPlugin, PluginContext


class SubAgentPlugin(AgentPlugin):
    """子代理机制插件"""
    
    name = "subagent"
    priority = 80
    
    def __init__(self, config=None):
        super().__init__(config)
        self._task_tool: Optional[TaskTool] = None
    
    def _initialize(self) -> None:
        if not self.config.subagent_enabled or not self.context.tool_registry:
            return
        
        self._register_task_tool()
    
    def _register_task_tool(self) -> None:
        """注册 TaskTool"""
        
        def agent_factory(agent_type: str):
            """子代理工厂函数"""
            # 决定使用哪个 LLM
            if self.config.subagent_use_light_llm:
                light_llm = self._create_light_llm()
            else:
                light_llm = self.context.llm
            
            return default_subagent_factory(
                agent_type=agent_type,
                llm=light_llm,
                tool_registry=self.context.tool_registry,
                config=self.config
            )
        
        self._task_tool = TaskTool(
            agent_factory=agent_factory,
            tool_registry=self.context.tool_registry,
            config=self.config
        )
        
        self.context.tool_registry.register_tool(self._task_tool)
    
    def _create_light_llm(self):
        """创建轻量模型 LLM 实例"""
        from ..core.llm import HelloAgentsLLM
        
        light_llm = HelloAgentsLLM(
            provider=self.config.subagent_light_llm_provider,
            model=self.config.subagent_light_llm_model,
            temperature=self.context.llm.temperature if hasattr(self.context.llm, 'temperature') else 0.7,
            max_tokens=self.context.llm.max_tokens if hasattr(self.context.llm, 'max_tokens') else None
        )
        return light_llm
    
    # ===== 子代理执行 API =====
    
    def run_as_subagent(
        self,
        task: str,
        tool_filter: Optional[ToolFilter] = None,
        return_summary: bool = True,
        max_steps_override: Optional[int] = None
    ) -> Dict[str, Any]:
        """作为子代理运行（上下文隔离模式）"""
        import time
        from datetime import datetime
        
        # 1. 保存当前状态
        history_plugin = self.context.agent._plugin_manager.get_plugin("history")
        original_history = history_plugin.get_history().copy() if history_plugin else []
        
        original_tools = None
        original_max_steps = None
        
        # 2. 创建隔离的新历史
        if history_plugin:
            history_plugin.clear_history()
        
        # 3. 应用工具过滤
        if tool_filter and self.context.tool_registry:
            original_tools = self._apply_tool_filter(tool_filter)
        
        # 4. 覆盖最大步数
        if max_steps_override is not None and hasattr(self.context.agent, 'max_steps'):
            original_max_steps = self.context.agent.max_steps
            self.context.agent.max_steps = max_steps_override
        
        start_time = time.time()
        success = False
        result = ""
        error_msg = None
        
        try:
            # 5. 执行任务
            result = self.context.agent.run(task)
            success = True
        
        except KeyboardInterrupt:
            error_msg = "用户中断"
            raise
        
        except Exception as e:
            error_msg = str(e)
            result = f"执行失败: {error_msg}"
        
        finally:
            duration = time.time() - start_time
            metadata = self._get_subagent_metadata(duration, error_msg)
            
            # 生成摘要
            if return_summary:
                summary = self._generate_subagent_summary(task, result, metadata)
            
            # 6. 恢复原始状态
            if history_plugin:
                history_plugin.clear_history()
                for msg in original_history:
                    history_plugin.add_message(msg)
            
            if original_tools is not None:
                self._restore_tools(original_tools)
            
            if original_max_steps is not None:
                self.context.agent.max_steps = original_max_steps
        
        # 返回结果
        if return_summary:
            return {
                "success": success,
                "summary": summary,
                "metadata": metadata
            }
        else:
            return {
                "success": success,
                "result": result,
                "metadata": metadata
            }
    
    def _apply_tool_filter(self, tool_filter: ToolFilter) -> List[str]:
        if not self.context.tool_registry:
            return []
        
        original_tools = self.context.tool_registry.list_tools()
        filtered_tools = tool_filter.filter(original_tools)
        
        for tool_name in original_tools:
            if tool_name not in filtered_tools:
                self.context.tool_registry._temp_disabled_tools = getattr(
                    self.context.tool_registry, '_temp_disabled_tools', {}
                )
                tool = self.context.tool_registry.get_tool(tool_name)
                if tool:
                    self.context.tool_registry._temp_disabled_tools[tool_name] = tool
                    if tool_name in self.context.tool_registry._tools:
                        del self.context.tool_registry._tools[tool_name]
        
        return original_tools
    
    def _restore_tools(self, original_tools: List[str]):
        if not self.context.tool_registry:
            return
        
        if hasattr(self.context.tool_registry, '_temp_disabled_tools'):
            for tool_name, tool in self.context.tool_registry._temp_disabled_tools.items():
                self.context.tool_registry._tools[tool_name] = tool
            self.context.tool_registry._temp_disabled_tools = {}
    
    def _get_subagent_metadata(self, duration: float, error: Optional[str]) -> Dict[str, Any]:
        history_plugin = self.context.agent._plugin_manager.get_plugin("history")
        history = history_plugin.get_history() if history_plugin else []
        
        steps = sum(1 for msg in history if msg.role == "assistant")
        
        total_chars = sum(len(msg.content) for msg in history)
        tokens = total_chars // 4
        
        tools_used = self._extract_tools_from_history(history)
        
        metadata = {
            "steps": steps,
            "tokens": tokens,
            "duration_seconds": round(duration, 2),
            "tools_used": tools_used
        }
        
        if error:
            metadata["error"] = error
        
        return metadata
    
    def _extract_tools_from_history(self, history: List) -> List[str]:
        tools = set()
        
        for msg in history:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if isinstance(tool_call, dict) and 'function' in tool_call:
                        tools.add(tool_call['function'].get('name', ''))
            
            if msg.role == "assistant" and "Action:" in msg.content:
                import re
                matches = re.findall(r'Action:\s*(\w+)\[', msg.content)
                tools.update(matches)
        
        return sorted(list(tools))
    
    def _generate_subagent_summary(
        self,
        task: str,
        result: str,
        metadata: Dict[str, Any]
    ) -> str:
        max_result_len = 500
        if len(result) > max_result_len:
            result_preview = result[:max_result_len] + "..."
        else:
            result_preview = result
        
        summary_parts = [
            f"任务: {task}",
            f"结果: {result_preview}",
            f"步数: {metadata['steps']}",
            f"耗时: {metadata['duration_seconds']}秒"
        ]
        
        if metadata.get('tools_used'):
            summary_parts.append(f"工具: {', '.join(metadata['tools_used'])}")
        
        if metadata.get('error'):
            summary_parts.append(f"错误: {metadata['error']}")
        
        return "\n".join(summary_parts)