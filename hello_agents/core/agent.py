"""Agent基类"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING, AsyncGenerator
import asyncio
from .message import Message
from .llm import HelloAgentsLLM
from .config import Config
from .lifecycle import AgentEvent, EventType, LifecycleHook, ExecutionContext
from .plugins import PluginManager, PluginContext, create_default_plugins

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry
    from ..observability.trace_logger import TraceLogger
    from ..tools.tool_filter import ToolFilter
    from ..context.history import HistoryManager
    from ..context.token_counter import TokenCounter
    from ..context.truncator import ObservationTruncator
    from ..core.session_store import SessionStore
    from ..skills.loader import SkillLoader


class Agent(ABC):
    """Agent基类

    集成能力：
    - HistoryManager: 历史管理与压缩
    - ObservationTruncator: 工具输出截断
    - TraceLogger: 可观测性（JSONL + HTML）
    - ToolRegistry: 工具管理（可选）
    - SkillLoader: 知识外化（可选）

    向后兼容：
    - self._history 属性仍然可用（通过 property 代理）
    - add_message/clear_history/get_history 方法保持不变
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional['ToolRegistry'] = None
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()

        # 工具注册表（可选）
        self.tool_registry = tool_registry

        # 插件系统初始化
        self._plugin_manager = PluginManager()
        self._init_plugins()

        # 向后兼容属性（代理到插件）
        self._init_compat_properties()

    def _init_plugins(self) -> None:
        """初始化插件系统"""
        # 创建插件上下文
        context = PluginContext(
            agent=self,
            config=self.config,
            llm=self.llm,
            tool_registry=self.tool_registry
        )
        
        # 创建并注册默认插件
        plugins = create_default_plugins(self.config, self.tool_registry)
        self._plugin_manager.register_many(plugins)
        
        # 初始化所有插件
        self._plugin_manager.initialize_all(context)

    def _init_compat_properties(self) -> None:
        """初始化向后兼容属性（代理到插件）"""
        # 这些属性将通过 property 动态获取
        pass

    @property
    def _plugin_manager(self) -> PluginManager:
        return self.__dict__.get('_plugin_manager')

    @_plugin_manager.setter
    def _plugin_manager(self, value: PluginManager):
        self.__dict__['_plugin_manager'] = value

    def _get_plugin(self, name: str):
        """获取插件实例"""
        return self._plugin_manager.get_plugin(name)

    @property
    def history_manager(self):
        """向后兼容：历史管理器"""
        plugin = self._get_plugin("history")
        return plugin.history_manager if plugin else None

    @property
    def token_counter(self):
        """向后兼容：Token 计数器"""
        plugin = self._get_plugin("token_counter")
        return plugin.token_counter if plugin else None

    @property
    def truncator(self):
        """向后兼容：截断器"""
        plugin = self._get_plugin("truncator")
        return plugin.truncator if plugin else None

    @property
    def trace_logger(self):
        """向后兼容：追踪记录器"""
        plugin = self._get_plugin("trace")
        return plugin.trace_logger if plugin else None

    @property
    def skill_loader(self):
        """向后兼容：技能加载器"""
        plugin = self._get_plugin("skill")
        return plugin.skill_loader if plugin else None

    @property
    def session_store(self):
        """向后兼容：会话存储"""
        plugin = self._get_plugin("session")
        return plugin.session_store if plugin else None

    @property
    def _session_metadata(self):
        """向后兼容：会话元数据"""
        return self._plugin_manager.context.session_metadata if self._plugin_manager else {}

    @_session_metadata.setter
    def _session_metadata(self, value):
        if self._plugin_manager:
            self._plugin_manager.context.session_metadata = value

    @property
    def _start_time(self):
        """向后兼容：开始时间"""
        return self._plugin_manager.context.start_time if self._plugin_manager else None

    @_start_time.setter
    def _start_time(self, value):
        if self._plugin_manager:
            self._plugin_manager.context.start_time = value

    @property
    def _history_token_count(self) -> int:
        """向后兼容：历史 Token 计数（代理到 TokenCounterPlugin）"""
        token_plugin = self._get_plugin("token_counter")
        return token_plugin.total_tokens if token_plugin else 0

    @_history_token_count.setter
    def _history_token_count(self, value: int):
        token_plugin = self._get_plugin("token_counter")
        if token_plugin:
            token_plugin._total_tokens = value

    @property
    def _history(self) -> List[Message]:
        """向后兼容：通过 property 代理到 HistoryManager"""
        if self.history_manager:
            return self.history_manager.get_history()
        return []

    @_history.setter
    def _history(self, value: List[Message]):
        """向后兼容：允许直接设置历史"""
        if self.history_manager:
            self.history_manager.clear()
            for msg in value:
                self.history_manager.append(msg)

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent（同步版本）"""
        pass

    # ==================== 异步生命周期方法 ====================

    async def arun(
        self,
        input_text: str,
        on_start: LifecycleHook = None,
        on_step: LifecycleHook = None,
        on_finish: LifecycleHook = None,
        on_error: LifecycleHook = None,
        **kwargs
    ) -> str:
        """
        异步执行 Agent（真正的异步实现）

        支持：
        - 并发工具执行（通过 max_concurrent_tools 控制）
        - 完整的生命周期钩子
        - 正确的错误处理

        Args:
            input_text: 输入文本
            on_start: Agent 开始执行时的钩子
            on_step: 每个推理步骤的钩子
            on_finish: Agent 执行完成时的钩子
            on_error: 发生错误时的钩子
            **kwargs: 其他参数

        Returns:
            执行结果

        Example:
            >>> agent = SimpleAgent(...)
            >>> result = await agent.arun("Hello", on_start=my_hook)
        """
        # 触发开始事件
        await self._emit_event(
            EventType.AGENT_START,
            on_start,
            input_text=input_text
        )

        try:
            # 调用子类实现的异步核心逻辑
            result = await self._arun_impl(input_text, on_step, **kwargs)

            # 触发完成事件
            await self._emit_event(
                EventType.AGENT_FINISH,
                on_finish,
                result=result
            )

            return result

        except Exception as e:
            # 触发错误事件
            await self._emit_event(
                EventType.AGENT_ERROR,
                on_error,
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def _arun_impl(self, input_text: str, on_step: LifecycleHook = None, **kwargs) -> str:
        """异步核心实现 - 子类必须重写此方法
        
        默认实现：在线程池中运行同步 run() 方法（向后兼容）
        子类应该重写此方法实现真正的异步逻辑（如工具并行）
        
        Args:
            input_text: 输入文本
            on_step: 步骤钩子
            **kwargs: 其他参数
            
        Returns:
            执行结果
        """
        # 默认实现：在线程池中运行同步 run()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.run(input_text, **kwargs)
        )

    async def arun_stream(
        self,
        input_text: str,
        on_start: LifecycleHook = None,
        on_step: LifecycleHook = None,
        on_finish: LifecycleHook = None,
        on_error: LifecycleHook = None,
        **kwargs
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        真正的流式异步执行 Agent

        逐步产出 AgentEvent 事件，支持：
        - 实时 LLM 流式输出 (LLM_CHUNK)
        - 工具调用进度 (TOOL_CALL, TOOL_RESULT)
        - 步骤级事件 (STEP_START, STEP_FINISH)
        - 生命周期事件 (AGENT_START, AGENT_FINISH, AGENT_ERROR)

        Args:
            input_text: 输入文本
            on_start: 开始钩子
            on_step: 步骤钩子
            on_finish: 完成钩子
            on_error: 错误钩子
            **kwargs: 其他参数

        Yields:
            AgentEvent: 生命周期事件流

        Example:
            >>> async for event in agent.arun_stream("Hello"):
            ...     print(event.type, event.data)
        """
        # 发送开始事件
        start_event = AgentEvent.create(
            EventType.AGENT_START,
            self.name,
            input_text=input_text
        )
        yield start_event
        
        if on_start:
            try:
                await on_start(start_event)
            except Exception:
                pass

        try:
            # 调用子类实现的流式核心逻辑
            async for event in self._arun_stream_impl(input_text, on_step, **kwargs):
                yield event
                if on_step and event.type in (EventType.STEP_START, EventType.STEP_FINISH, EventType.TOOL_CALL, EventType.TOOL_RESULT, EventType.LLM_CHUNK):
                    try:
                        await on_step(event)
                    except Exception:
                        pass

            # 发送完成事件
            finish_event = AgentEvent.create(
                EventType.AGENT_FINISH,
                self.name,
                result=event.data.get("result", "") if hasattr(event, 'data') else ""
            )
            yield finish_event
            
            if on_finish:
                try:
                    await on_finish(finish_event)
                except Exception:
                    pass

        except Exception as e:
            # 发送错误事件
            error_event = AgentEvent.create(
                EventType.AGENT_ERROR,
                self.name,
                error=str(e),
                error_type=type(e).__name__
            )
            yield error_event
            
            if on_error:
                try:
                    await on_error(error_event)
                except Exception:
                    pass
            raise

    async def _arun_stream_impl(self, input_text: str, on_step: LifecycleHook = None, **kwargs) -> AsyncGenerator[AgentEvent, None]:
        """流式异步核心实现 - 子类必须重写此方法
        
        默认实现：运行 _arun_impl 并产出开始/完成事件
        子类应该重写此方法实现真正的流式输出（逐 token、工具流式等）
        
        Args:
            input_text: 输入文本
            on_step: 步骤钩子
            **kwargs: 其他参数
            
        Yields:
            AgentEvent: 生命周期事件流
        """
        # 默认实现：执行 _arun_impl 并产出事件
        result = await self._arun_impl(input_text, on_step, **kwargs)
        
        # 产出完成事件（包含结果）
        yield AgentEvent.create(
            EventType.AGENT_FINISH,
            self.name,
            result=result
        )

    async def _emit_event(
        self,
        event_type: EventType,
        hook: LifecycleHook,
        **data
    ):
        """触发事件并调用钩子

        Args:
            event_type: 事件类型
            hook: 生命周期钩子（可选）
            **data: 事件数据
        """
        event = AgentEvent.create(event_type, self.name, **data)

        if hook:
            try:
                # 使用 asyncio.wait_for 设置超时
                timeout = getattr(self.config, 'hook_timeout_seconds', 5.0)
                await asyncio.wait_for(hook(event), timeout=timeout)
            except asyncio.TimeoutError:
                # 钩子超时不应中断主流程
                if hasattr(self, 'trace_logger') and self.trace_logger:
                    self.trace_logger.log_event(
                        "hook_timeout",
                        {"event_type": event_type.value, "timeout": timeout}
                    )
            except Exception as e:
                # 钩子异常不应中断主流程
                if hasattr(self, 'trace_logger') and self.trace_logger:
                    self.trace_logger.log_event(
                        "hook_error",
                        {"event_type": event_type.value, "error": str(e)}
                    )

    def add_message(self, message: Message):
        """添加消息到历史记录（代理到 HistoryPlugin）"""
        history_plugin = self._get_plugin("history")
        if history_plugin:
            history_plugin.add_message(message)

    def clear_history(self):
        """清空历史记录（代理到 HistoryPlugin + TokenCounterPlugin）"""
        history_plugin = self._get_plugin("history")
        if history_plugin:
            history_plugin.clear_history()
        token_plugin = self._get_plugin("token_counter")
        if token_plugin:
            token_plugin.reset()

    def get_history(self) -> List[Message]:
        """获取历史记录"""
        if self.history_manager:
            return self.history_manager.get_history()
        return []

    def _should_compress(self) -> bool:
        """判断是否需要压缩历史（代理到 HistoryPlugin）"""
        history_plugin = self._get_plugin("history")
        if history_plugin:
            return history_plugin._should_compress()
        return False

    def _compress_history(self):
        """压缩历史（代理到 HistoryPlugin）"""
        history_plugin = self._get_plugin("history")
        if history_plugin:
            history_plugin._compress_history()

    def _generate_simple_summary(self, history: List[Message]) -> str:
        """生成简单摘要（代理到 HistoryPlugin）"""
        history_plugin = self._get_plugin("history")
        if history_plugin:
            return history_plugin._generate_simple_summary(history)
        return ""

    def _generate_smart_summary(self, history: List[Message]) -> str:
        """生成智能摘要（内部实现，避免循环引用）"""
        # 1. 提取要压缩的历史片段
        if self.history_manager:
            boundaries = self.history_manager.find_round_boundaries()
        else:
            return self._generate_simple_summary(history)
        
        if len(boundaries) <= self.config.min_retain_rounds:
            return self._generate_simple_summary(history)

        # 保留最近 N 轮，压缩之前的
        keep_from_index = boundaries[-self.config.min_retain_rounds]
        to_compress = history[:keep_from_index]

        if not to_compress:
            return self._generate_simple_summary(history)

        # 2. 构建摘要 Prompt
        history_text = self._format_history_for_summary(to_compress)

        summary_prompt = f"""请将以下对话历史压缩为结构化摘要，保留关键信息：

## 对话历史
{history_text}

## 摘要要求
1. **任务目标**：用户想要完成什么？
2. **关键决策**：做了哪些重要决定？
3. **已完成工作**：完成了哪些任务？（列表形式）
4. **待处理事项**：还有什么未完成？
5. **重要发现**：有哪些关键信息或问题？

请用简洁的中文输出，每部分不超过 3 行。"""

        # 3. 调用轻量 LLM（节省成本）
        try:
            summary_llm = self._get_summary_llm()

            messages = [
                {"role": "system", "content": "你是一个专业的对话摘要助手，擅长提取关键信息。"},
                {"role": "user", "content": summary_prompt}
            ]

            # 非流式调用，快速获取结果
            summary = summary_llm.invoke(
                messages,
                temperature=self.config.summary_temperature,
                max_tokens=self.config.summary_max_tokens
            )

            return f"""## 历史摘要（{len(to_compress)} 条消息）
{summary}

---
（已压缩，保留最近 {self.config.min_retain_rounds} 轮完整对话）"""

        except Exception as e:
            # 回退到简单摘要
            print(f"⚠️ 智能摘要生成失败: {e}，使用简单摘要")
            return self._generate_simple_summary(history)

    def _format_history_for_summary(self, history: List[Message]) -> str:
        """格式化历史消息用于摘要生成"""
        history_plugin = self._get_plugin("history")
        if history_plugin and hasattr(history_plugin, '_format_history_for_summary'):
            return history_plugin._format_history_for_summary(history)
        return ""

    def _get_summary_llm(self):
        """获取摘要专用 LLM（轻量模型）"""
        # 保持原有逻辑，因为它依赖 Agent 实例
        if not hasattr(self, '_summary_llm'):
            from ..core.llm import HelloAgentsLLM

            provider = self.config.summary_llm_provider
            model = self.config.summary_llm_model

            self._summary_llm = HelloAgentsLLM(
                provider=provider,
                model=model,
                temperature=self.config.summary_temperature,
                max_tokens=self.config.summary_max_tokens
            )

        return self._summary_llm

    def __str__(self) -> str:
        return f"Agent(name={self.name}, model={self.llm.model})"

    def __repr__(self) -> str:
        return self.__str__()

    # ==================== 工具调用通用能力（从 FunctionCallAgent 提取）====================

    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        """构建工具 JSON Schema

        统一的工具 schema 构建逻辑，支持：
        - Tool 对象（带参数定义）
        - 函数工具（简化注册）

        Returns:
            工具 schema 列表
        """
        if not self.tool_registry:
            return []

        schemas: List[Dict[str, Any]] = []

        # 1. 处理 Tool 对象
        for tool in self.tool_registry.get_all_tools():
            properties: Dict[str, Any] = {}
            required: List[str] = []

            try:
                parameters = tool.get_parameters()
            except Exception:
                parameters = []

            for param in parameters:
                properties[param.name] = {
                    "type": self._map_parameter_type(param.type),
                    "description": param.description or ""
                }
                if param.default is not None:
                    properties[param.name]["default"] = param.default
                if getattr(param, "required", True):
                    required.append(param.name)

            schema: Dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties
                    }
                }
            }
            if required:
                schema["function"]["parameters"]["required"] = required
            schemas.append(schema)

        # 2. 处理函数工具
        function_map = getattr(self.tool_registry, "_functions", {})
        for name, info in function_map.items():
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": info.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "输入文本"
                            }
                        },
                        "required": ["input"]
                    }
                }
            })

        return schemas

    @staticmethod
    def _map_parameter_type(param_type: str) -> str:
        """将工具参数类型映射为 JSON Schema 允许的类型

        Args:
            param_type: 工具参数类型

        Returns:
            JSON Schema 类型
        """
        normalized = (param_type or "").lower()
        if normalized in {"string", "number", "integer", "boolean", "array", "object"}:
            return normalized
        return "string"

    def _convert_parameter_types(self, tool_name: str, param_dict: Dict[str, Any]) -> Dict[str, Any]:
        """根据工具定义转换参数类型

        Args:
            tool_name: 工具名称
            param_dict: 参数字典

        Returns:
            类型转换后的参数字典
        """
        if not self.tool_registry:
            return param_dict

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return param_dict

        try:
            tool_params = tool.get_parameters()
        except Exception:
            return param_dict

        type_mapping = {param.name: param.type for param in tool_params}
        converted: Dict[str, Any] = {}

        for key, value in param_dict.items():
            param_type = type_mapping.get(key)
            if not param_type:
                converted[key] = value
                continue

            try:
                normalized = param_type.lower()
                if normalized in {"number", "float"}:
                    converted[key] = float(value)
                elif normalized in {"integer", "int"}:
                    converted[key] = int(value)
                elif normalized in {"boolean", "bool"}:
                    if isinstance(value, bool):
                        converted[key] = value
                    elif isinstance(value, (int, float)):
                        converted[key] = bool(value)
                    elif isinstance(value, str):
                        converted[key] = value.lower() in {"true", "1", "yes"}
                    else:
                        converted[key] = bool(value)
                else:
                    converted[key] = value
            except (TypeError, ValueError):
                converted[key] = value

        return converted

    def _execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """执行工具调用并返回字符串结果

        统一的工具执行逻辑，支持：
        - Tool 对象（带类型转换）
        - 函数工具（简化调用）

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果（字符串格式）
        """
        if not self.tool_registry:
            return "❌ 错误：未配置工具注册表"

        # 1. 尝试执行 Tool 对象
        tool = self.tool_registry.get_tool(tool_name)
        if tool:
            try:
                typed_arguments = self._convert_parameter_types(tool_name, arguments)
                response = tool.run_with_timing(typed_arguments)

                # 根据状态添加前缀
                from ..tools.response import ToolStatus
                if response.status == ToolStatus.ERROR:
                    error_code = response.error_info.get("code", "UNKNOWN") if response.error_info else "UNKNOWN"
                    return f"❌ 错误 [{error_code}]: {response.text}"
                elif response.status == ToolStatus.PARTIAL:
                    return f"⚠️ 部分成功: {response.text}"
                else:
                    return response.text
            except Exception as exc:
                return f"❌ 工具调用失败：{exc}"

        # 2. 尝试执行函数工具
        func = self.tool_registry.get_function(tool_name)
        if func:
            try:
                input_text = arguments.get("input", "")
                response = self.tool_registry.execute_tool(tool_name, input_text)

                # 根据状态添加前缀
                from ..tools.response import ToolStatus
                if response.status == ToolStatus.ERROR:
                    error_code = response.error_info.get("code", "UNKNOWN") if response.error_info else "UNKNOWN"
                    return f"❌ 错误 [{error_code}]: {response.text}"
                elif response.status == ToolStatus.PARTIAL:
                    return f"⚠️ 部分成功: {response.text}"
                else:
                    return response.text
            except Exception as exc:
                return f"❌ 工具调用失败：{exc}"

        return f"❌ 错误：未找到工具 '{tool_name}'"

    # ==================== 会话持久化能力 ====================

    def _auto_save(self):
        """自动保存会话（静默失败，代理到 SessionPlugin）"""
        session_plugin = self._get_plugin("session")
        if session_plugin:
            session_plugin.maybe_auto_save()

    def save_session(self, session_name: str) -> str:
        """手动保存会话（代理到 SessionPlugin）"""
        session_plugin = self._get_plugin("session")
        if session_plugin:
            return session_plugin.save_session(session_name)
        raise RuntimeError("会话持久化未启用，请在 Config 中设置 session_enabled=True")

    def load_session(self, filepath: str, check_consistency: bool = True) -> None:
        """加载会话（代理到 SessionPlugin）"""
        session_plugin = self._get_plugin("session")
        if session_plugin:
            session_plugin.load_session(filepath, check_consistency)
        else:
            raise RuntimeError("会话持久化未启用，请在 Config 中设置 session_enabled=True")

    def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有可用会话（代理到 SessionPlugin）"""
        session_plugin = self._get_plugin("session")
        if session_plugin:
            return session_plugin.list_sessions()
        return []

    def _get_agent_config(self) -> Dict[str, Any]:
        """获取 Agent 配置信息"""
        config = {
            "name": self.name,
            "agent_type": self.__class__.__name__,
            "llm_provider": getattr(self.llm, 'provider', 'unknown'),
            "llm_model": getattr(self.llm, 'model_id', getattr(self.llm, 'model', 'unknown'))
        }

        if hasattr(self, 'max_steps'):
            config["max_steps"] = self.max_steps

        return config

    def _compute_tool_schema_hash(self) -> str:
        """计算工具 Schema 哈希（代理到 SessionPlugin）"""
        session_plugin = self._get_plugin("session")
        if session_plugin:
            return session_plugin._compute_tool_schema_hash()
        if not self.tool_registry:
            return "no-tools"
        
        import json
        from hashlib import sha256
        tools_signature = {}
        for tool_name in sorted(self.tool_registry.list_tools()):
            tool = self.tool_registry.get_tool(tool_name)
            if tool:
                tools_signature[tool_name] = {
                    "name": tool.name,
                    "description": tool.description[:100] if tool.description else "",
                    "parameters": list(tool.parameters.keys()) if hasattr(tool, 'parameters') and tool.parameters else []
                }
        schema_str = json.dumps(tools_signature, sort_keys=True)
        return sha256(schema_str.encode()).hexdigest()[:16]

    def _get_read_cache(self) -> Dict[str, Dict]:
        """获取 Read 工具的元数据缓存"""
        if self.tool_registry and hasattr(self.tool_registry, 'read_metadata_cache'):
            return self.tool_registry.read_metadata_cache
        return {}

    # ==================== 子代理机制 ====================

    def run_as_subagent(
        self,
        task: str,
        tool_filter: Optional['ToolFilter'] = None,
        return_summary: bool = True,
        max_steps_override: Optional[int] = None
    ) -> Dict[str, Any]:
        """作为子代理运行（代理到 SubAgentPlugin）"""
        subagent_plugin = self._get_plugin("subagent")
        if subagent_plugin:
            return subagent_plugin.run_as_subagent(task, tool_filter, return_summary, max_steps_override)
        
        # 如果插件未启用，但用户显式调用，尝试临时创建并执行
        # 这支持测试场景：config 禁用自动注册但手动调用
        from ..tools.tool_filter import ToolFilter
        from ..agents.factory import default_subagent_factory
        from ..tools.builtin.task_tool import TaskTool
        import time
        
        # 创建临时工厂
        def agent_factory(agent_type: str):
            if self.config.subagent_use_light_llm:
                light_llm = self._create_light_llm()
            else:
                light_llm = self.llm
            return default_subagent_factory(
                agent_type=agent_type,
                llm=light_llm,
                tool_registry=self.tool_registry,
                config=self.config
            )
        
        # 保存状态
        history_plugin = self._get_plugin("history")
        original_history = history_plugin.get_history().copy() if history_plugin else []
        original_tools = None
        original_max_steps = None
        
        if history_plugin:
            history_plugin.clear_history()
        
        if tool_filter and self.tool_registry:
            original_tools = self._apply_tool_filter(tool_filter)
        
        if max_steps_override is not None and hasattr(self, 'max_steps'):
            original_max_steps = self.max_steps
            self.max_steps = max_steps_override
        
        start_time = time.time()
        success = False
        result = ""
        error_msg = None
        
        try:
            result = self.run(task)
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
            
            if return_summary:
                summary = self._generate_subagent_summary(task, result, metadata)
            
            if history_plugin:
                history_plugin.clear_history()
                for msg in original_history:
                    history_plugin.add_message(msg)
            
            if original_tools is not None:
                self._restore_tools(original_tools)
            
            if original_max_steps is not None:
                self.max_steps = original_max_steps
        
        if return_summary:
            return {"success": success, "summary": summary, "metadata": metadata}
        else:
            return {"success": success, "result": result, "metadata": metadata}

    def _apply_tool_filter(self, tool_filter: 'ToolFilter') -> List[str]:
        """应用工具过滤器

        Args:
            tool_filter: 工具过滤器实例

        Returns:
            原始工具列表（用于恢复）
        """
        if not self.tool_registry:
            return []

        # 保存原始工具列表
        original_tools = self.tool_registry.list_tools()

        # 获取过滤后的工具列表
        filtered_tools = tool_filter.filter(original_tools)

        # 临时移除不允许的工具
        for tool_name in original_tools:
            if tool_name not in filtered_tools:
                self.tool_registry._temp_disabled_tools = getattr(
                    self.tool_registry, '_temp_disabled_tools', {}
                )
                tool = self.tool_registry.get_tool(tool_name)
                if tool:
                    self.tool_registry._temp_disabled_tools[tool_name] = tool
                    # 从注册表中临时移除
                    if tool_name in self.tool_registry._tools:
                        del self.tool_registry._tools[tool_name]

        return original_tools

    def _restore_tools(self, original_tools: List[str]):
        """恢复原始工具列表

        Args:
            original_tools: 原始工具名称列表
        """
        if not self.tool_registry:
            return

        # 恢复被禁用的工具
        if hasattr(self.tool_registry, '_temp_disabled_tools'):
            for tool_name, tool in self.tool_registry._temp_disabled_tools.items():
                self.tool_registry._tools[tool_name] = tool

            # 清空临时禁用列表
            self.tool_registry._temp_disabled_tools = {}

    def _get_subagent_metadata(self, duration: float, error: Optional[str]) -> Dict[str, Any]:
        """获取子代理执行元数据

        Args:
            duration: 执行时长（秒）
            error: 错误信息（可选）

        Returns:
            元数据字典
        """
        history = self.history_manager.get_history()

        # 估算步数（用户+助手消息对）
        steps = sum(1 for msg in history if msg.role == "assistant")

        # 估算 Token 数（简化：字符数 / 4）
        total_chars = sum(len(msg.content) for msg in history)
        tokens = total_chars // 4

        # 提取使用的工具
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

    def _extract_tools_from_history(self, history: List[Message]) -> List[str]:
        """从历史中提取使用的工具

        Args:
            history: 历史消息列表

        Returns:
            工具名称列表（去重）
        """
        tools = set()

        for msg in history:
            # 检查 tool_calls（FunctionCallAgent）
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if isinstance(tool_call, dict) and 'function' in tool_call:
                        tools.add(tool_call['function'].get('name', ''))

            # 检查内容中的工具调用（ReActAgent）
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
        """生成子代理执行摘要

        Args:
            task: 任务描述
            result: 执行结果
            metadata: 执行元数据

        Returns:
            摘要文本
        """
        # 截断结果（避免摘要过长）
        max_result_len = 500
        if len(result) > max_result_len:
            result_preview = result[:max_result_len] + "..."
        else:
            result_preview = result

        # 构建摘要
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

    def _register_task_tool(self):
        """注册 TaskTool（子代理工具）

        自动注册逻辑，支持用户自定义工厂函数。
        """
        from ..tools.builtin.task_tool import TaskTool
        from ..agents.factory import default_subagent_factory

        # 创建子代理工厂函数
        def agent_factory(agent_type: str) -> Agent:
            """子代理工厂函数"""
            # 决定使用哪个 LLM
            if self.config.subagent_use_light_llm:
                # 使用轻量模型
                light_llm = self._create_light_llm()
            else:
                # 使用主模型
                light_llm = self.llm

            # 使用默认工厂创建子代理
            return default_subagent_factory(
                agent_type=agent_type,
                llm=light_llm,
                tool_registry=self.tool_registry,
                config=self.config
            )
