"""ToolPlugin - 工具系统插件

职责：工具注册表管理、工具 Schema 构建、工具执行、参数类型转换
"""

from typing import List, Dict, Any, Optional
from ..tools.registry import ToolRegistry
from ..tools.response import ToolResponse, ToolStatus
from ..tools.errors import ToolErrorCode
from .plugins import AgentPlugin, PluginContext


class ToolPlugin(AgentPlugin):
    """工具系统插件"""
    
    name = "tool"
    priority = 30
    
    def __init__(self, config=None, tool_registry: Optional[ToolRegistry] = None):
        super().__init__(config)
        self._tool_registry = tool_registry
        self._tool_schemas_cache: Optional[List[Dict[str, Any]]] = None
    
    def _initialize(self) -> None:
        if self._tool_registry is None:
            self._tool_registry = ToolRegistry()
        
        self.context.tool_registry = self._tool_registry
        
        # 如果 Agent 有自定义工厂，在此注册内置工具
        self._register_builtin_tools()
    
    @property
    def tool_registry(self) -> ToolRegistry:
        return self._tool_registry
    
    def _register_builtin_tools(self) -> None:
        """注册内置工具（根据配置）"""
        # 这里不注册具体工具，由各功能插件（SubAgentPlugin, TodoPlugin 等）负责
        # 此处只确保工具注册表存在
        pass
    
    # ===== 向后兼容 API =====
    
    def build_tool_schemas(self) -> List[Dict[str, Any]]:
        """构建工具 JSON Schema（缓存）"""
        if self._tool_schemas_cache is not None:
            return self._tool_schemas_cache
        
        if not self._tool_registry:
            return []
        
        schemas: List[Dict[str, Any]] = []
        
        # 1. 处理 Tool 对象
        for tool in self._tool_registry.get_all_tools():
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
        function_map = getattr(self._tool_registry, "_functions", {})
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
        
        self._tool_schemas_cache = schemas
        return schemas
    
    def invalidate_schema_cache(self) -> None:
        self._tool_schemas_cache = None
    
    @staticmethod
    def _map_parameter_type(param_type: str) -> str:
        normalized = (param_type or "").lower()
        if normalized in {"string", "number", "integer", "boolean", "array", "object"}:
            return normalized
        return "string"
    
    def convert_parameter_types(self, tool_name: str, param_dict: Dict[str, Any]) -> Dict[str, Any]:
        """根据工具定义转换参数类型"""
        if not self._tool_registry:
            return param_dict
        
        tool = self._tool_registry.get_tool(tool_name)
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
    
    def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """执行工具调用并返回字符串结果"""
        if not self._tool_registry:
            return "❌ 错误：未配置工具注册表"
        
        # 1. 尝试执行 Tool 对象
        tool = self._tool_registry.get_tool(tool_name)
        if tool:
            try:
                typed_arguments = self.convert_parameter_types(tool_name, arguments)
                response = tool.run_with_timing(typed_arguments)
                
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
        func = self._tool_registry.get_function(tool_name)
        if func:
            try:
                input_text = arguments.get("input", "")
                response = self._tool_registry.execute_tool(tool_name, input_text)
                
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