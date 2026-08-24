"""MCPPlugin - Model Context Protocol 支持插件

职责：
- 连接到 MCP 服务器
- 发现并注册 MCP 工具
- 代理工具调用到 MCP 服务器
- 支持多个 MCP 服务器连接
"""

from typing import List, Dict, Any, Optional, Callable
from .plugins import AgentPlugin, PluginContext
import asyncio
import json


class MCPClient:
    """MCP 客户端 - 连接单个 MCP 服务器"""
    
    def __init__(self, server_name: str, command: List[str], env: Optional[Dict[str, str]] = None):
        self.server_name = server_name
        self.command = command
        self.env = env or {}
        self._process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._tools: List[Dict[str, Any]] = []
        self._initialized = False
    
    async def connect(self) -> bool:
        """连接到 MCP 服务器"""
        try:
            # 启动服务器进程
            self._process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**import_os_environ(), **self.env}
            )
            
            # 启动读取循环
            asyncio.create_task(self._read_loop())
            
            # 发送初始化请求
            init_result = await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "hello-agents", "version": "1.0.0"}
            })
            
            if init_result:
                self._initialized = True
                # 列出工具
                await self.list_tools()
                return True
            return False
            
        except Exception as e:
            print(f"❌ MCP 连接失败 ({self.server_name}): {e}")
            return False
    
    async def _read_loop(self):
        """读取服务器响应循环"""
        if not self._process or not self._process.stdout:
            return
            
        async for line in self._process.stdout:
            line = line.decode().strip()
            if not line:
                continue
            try:
                response = json.loads(line)
                await self._handle_response(response)
            except json.JSONDecodeError:
                pass
    
    async def _handle_response(self, response: Dict[str, Any]):
        """处理服务器响应"""
        # 处理请求响应
        if "id" in response and response["id"] in self._pending_requests:
            future = self._pending_requests.pop(response["id"])
            if "error" in response:
                future.set_exception(Exception(response["error"].get("message", "Unknown error")))
            else:
                future.set_result(response.get("result"))
        
        # 处理通知
        elif "method" in response:
            method = response["method"]
            if method == "notifications/tools/list_changed":
                await self.list_tools()
    
    async def _send_request(self, method: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """发送 JSON-RPC 请求"""
        if not self._process or not self._process.stdin:
            return None
        
        self._request_id += 1
        request_id = self._request_id
        
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }
        
        future = asyncio.Future()
        self._pending_requests[request_id] = future
        
        try:
            request_line = json.dumps(request) + "\n"
            self._process.stdin.write(request_line.encode())
            await self._process.stdin.drain()
            
            # 等待响应（带超时）
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            print(f"⚠️ MCP 请求超时 ({self.server_name}): {method}")
            return None
        except Exception as e:
            self._pending_requests.pop(request_id, None)
            print(f"❌ MCP 请求失败 ({self.server_name}): {e}")
            return None
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """获取可用工具列表"""
        result = await self._send_request("tools/list")
        if result and "tools" in result:
            self._tools = result["tools"]
            # 添加服务器前缀避免冲突
            for tool in self._tools:
                tool["_mcp_server"] = self.server_name
        return self._tools
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """调用工具"""
        # 移除服务器前缀（如果有）
        original_name = name
        if name.startswith(f"{self.server_name}__"):
            original_name = name[len(f"{self.server_name}__"):]
        
        result = await self._send_request("tools/call", {
            "name": original_name,
            "arguments": arguments
        })
        
        if result:
            # 标准化返回格式
            if isinstance(result, dict) and "content" in result:
                # 处理 MCP 标准响应格式
                content = result["content"]
                if isinstance(content, list):
                    return "\n".join(
                        item.get("text", str(item)) 
                        for item in content 
                        if item.get("type") == "text"
                    )
                return str(content)
            return str(result)
        
        return f"❌ MCP 工具调用失败: {name}"
    
    async def disconnect(self):
        """断开连接"""
        if self._process:
            self._process.terminate()
            await self._process.wait()
            self._process = None
        self._initialized = False


def import_os_environ():
    import os
    return dict(os.environ)


class MCPPlugin(AgentPlugin):
    """MCP 支持插件 - 管理多个 MCP 服务器连接"""
    
    name = "mcp"
    priority = 25  # 在 ToolPlugin 之前加载，以便注册工具
    
    def __init__(self, config=None):
        super().__init__(config)
        self._clients: Dict[str, MCPClient] = {}
        self._mcp_config: List[Dict[str, Any]] = []
    
    def _initialize(self) -> None:
        # 从配置读取 MCP 服务器列表
        self._mcp_config = getattr(self.config, 'mcp_servers', [])
        
        if self._mcp_config:
            # 创建异步任务连接服务器
            # 注意：这里不能直接 await，需要在事件循环中运行
            # 使用 create_task 并在后台运行
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._connect_all())
                else:
                    loop.run_until_complete(self._connect_all())
            except RuntimeError:
                # 没有事件循环，稍后手动调用 connect
                pass
    
    async def _connect_all(self):
        """连接所有配置的 MCP 服务器"""
        for server_config in self._mcp_config:
            await self.add_server(server_config)
    
    async def add_server(self, server_config: Dict[str, Any]) -> bool:
        """添加并连接 MCP 服务器
        
        Args:
            server_config: {
                "name": "github",           # 服务器名称
                "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_TOKEN": "xxx"}  # 可选环境变量
            }
        """
        name = server_config.get("name")
        command = server_config.get("command")
        env = server_config.get("env")
        
        if not name or not command:
            print(f"❌ MCP 配置无效: {server_config}")
            return False
        
        client = MCPClient(name, command, env)
        success = await client.connect()
        
        if success:
            self._clients[name] = client
            # 注册工具到工具注册表
            await self._register_tools(client)
            print(f"✅ MCP 服务器已连接: {name} ({len(client._tools)} 个工具)")
            return True
        else:
            print(f"❌ MCP 服务器连接失败: {name}")
            return False
    
    async def _register_tools(self, client: MCPClient):
        """将 MCP 工具注册到本地工具注册表"""
        if not self.context.tool_registry:
            return
        
        from ..tools.base import Tool, ToolParameter
        
        for tool_def in client._tools:
            # 创建本地工具包装器
            tool_name = f"{client.server_name}__{tool_def['name']}"
            
            # 解析参数 schema
            parameters = []
            input_schema = tool_def.get("inputSchema", {})
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])
            
            for param_name, param_def in properties.items():
                parameters.append(ToolParameter(
                    name=param_name,
                    type=param_def.get("type", "string"),
                    description=param_def.get("description", ""),
                    required=param_name in required
                ))
            
            # 创建 MCP 工具
            mcp_tool = MCPTool(
                name=tool_name,
                description=tool_def.get("description", ""),
                parameters=parameters,
                mcp_client=client,
                original_name=tool_def["name"]
            )
            
            self.context.tool_registry.register_tool(mcp_tool)
    
    async def remove_server(self, name: str):
        """移除 MCP 服务器"""
        if name in self._clients:
            await self._clients[name].disconnect()
            del self._clients[name]
            
            # 从工具注册表移除相关工具
            if self.context.tool_registry:
                tools_to_remove = [
                    t for t in self.context.tool_registry.list_tools()
                    if t.startswith(f"{name}__")
                ]
                for tool_name in tools_to_remove:
                    self.context.tool_registry.unregister_tool(tool_name)
    
    def get_servers(self) -> List[str]:
        """获取已连接的服务器列表"""
        return list(self._clients.keys())
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """获取所有 MCP 工具"""
        tools = []
        for client in self._clients.values():
            tools.extend(client._tools)
        return tools
    
    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """调用 MCP 工具（带服务器前缀）"""
        # 解析服务器名称
        if "__" not in tool_name:
            return f"❌ 无效的 MCP 工具名格式: {tool_name} (应为 server__tool)"
        
        server_name, original_name = tool_name.split("__", 1)
        if server_name in self._clients:
            return await self._clients[server_name].call_tool(original_name, arguments)
        return f"❌ MCP 服务器未找到: {server_name}"
    
    async def teardown(self):
        """清理所有连接"""
        for client in self._clients.values():
            await client.disconnect()
        self._clients.clear()


class MCPTool:
    """MCP 工具包装器（不继承 Tool，避免循环导入）"""
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: List,
        mcp_client: MCPClient,
        original_name: str
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.mcp_client = mcp_client
        self.original_name = original_name
        self._expandable = False
    
    def get_parameters(self) -> List:
        """获取工具参数定义"""
        return self.parameters
    
    def run(self, arguments: Dict[str, Any]) -> str:
        """同步运行（不推荐，建议使用异步）"""
        # 创建新事件循环运行（仅用于向后兼容）
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.mcp_client.call_tool(self.original_name, arguments)
        )
    
    async def arun(self, arguments: Dict[str, Any]) -> str:
        """异步运行"""
        return await self.mcp_client.call_tool(self.original_name, arguments)
    
    def run_with_timing(self, arguments: Dict[str, Any]):
        """带计时的运行（返回 ToolResponse）"""
        import time
        from ..tools.response import ToolResponse
        
        start = time.time()
        try:
            # 运行异步方法
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(self.arun(arguments))
            elapsed = (time.time() - start) * 1000
            
            return ToolResponse.success(
                text=result,
                stats={"time_ms": elapsed}
            )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return ToolResponse.error(
                text=str(e),
                error_code="MCP_ERROR",
                error_message=str(e),
                stats={"time_ms": elapsed}
            )