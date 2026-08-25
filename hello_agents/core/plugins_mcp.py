"""MCPPlugin - Model Context Protocol 支持插件

职责：
- 连接到 MCP 服务器（支持连接池、健康检查、自动重连）
- 发现并注册 MCP 工具
- 代理工具调用到 MCP 服务器
- 支持多个 MCP 服务器连接
"""

from typing import List, Dict, Any, Optional, Callable
from .plugins import AgentPlugin, PluginContext
import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum


class ConnectionState(Enum):
    """连接状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    UNHEALTHY = "unhealthy"
    RECONNECTING = "reconnecting"


@dataclass
class MCPServerConfig:
    """MCP 服务器配置"""
    name: str
    command: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    # 连接池配置
    pool_size: int = 1
    # 健康检查配置
    health_check_interval: int = 30  # 秒
    health_check_timeout: int = 10   # 秒
    # 重连配置
    max_retries: int = 3
    retry_delay: float = 5.0         # 秒
    retry_backoff: float = 2.0       # 指数退避倍数
    # 超时配置
    request_timeout: float = 30.0
    connect_timeout: float = 10.0


class MCPConnectionPool:
    """MCP 连接池 - 管理单个服务器的多个连接"""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._connections: List[MCPClient] = []
        self._available: asyncio.Queue = asyncio.Queue()
        self._in_use: Set[MCPClient] = set()
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._state = ConnectionState.DISCONNECTED
        self._last_health_check: float = 0
        self._consecutive_failures = 0
    
    @property
    def state(self) -> ConnectionState:
        return self._state
    
    async def initialize(self) -> bool:
        """初始化连接池"""
        async with self._lock:
            if self._state != ConnectionState.DISCONNECTED:
                return True
            
            self._state = ConnectionState.CONNECTING
            
            # 创建初始连接
            for _ in range(self.config.pool_size):
                client = MCPClient(
                    f"{self.config.name}-{len(self._connections)}",
                    self.config.command,
                    self.config.env
                )
                if await client.connect():
                    self._connections.append(client)
                    await self._available.put(client)
                else:
                    await client.disconnect()
            
            if self._connections:
                self._state = ConnectionState.CONNECTED
                # 启动健康检查
                self._health_check_task = asyncio.create_task(self._health_check_loop())
                return True
            
            self._state = ConnectionState.DISCONNECTED
            return False
    
    async def acquire(self) -> Optional[MCPClient]:
        """获取可用连接"""
        if self._state != ConnectionState.CONNECTED:
            # 尝试重新初始化
            await self.initialize()
            if self._state != ConnectionState.CONNECTED:
                return None
        
        try:
            # 等待可用连接（带超时）
            client = await asyncio.wait_for(
                self._available.get(),
                timeout=self.config.request_timeout
            )
            self._in_use.add(client)
            return client
        except asyncio.TimeoutError:
            print(f"⚠️ MCP 连接池获取超时 ({self.config.name})")
            return None
    
    async def release(self, client: MCPClient):
        """释放连接回池"""
        if client in self._in_use:
            self._in_use.remove(client)
            # 检查连接是否仍然健康
            if await self._is_healthy(client):
                await self._available.put(client)
            else:
                # 连接不健康，创建新连接替换
                await self._replace_connection(client)
    
    async def _is_healthy(self, client: MCPClient) -> bool:
        """检查连接健康状态"""
        try:
            # 发送 ping 请求
            result = await asyncio.wait_for(
                client._send_request("ping"),
                timeout=self.config.health_check_timeout
            )
            return result is not None
        except Exception:
            return False
    
    async def _replace_connection(self, old_client: MCPClient):
        """替换失效连接"""
        await old_client.disconnect()
        if old_client in self._connections:
            self._connections.remove(old_client)
        
        # 创建新连接
        new_client = MCPClient(
            f"{self.config.name}-{len(self._connections)}",
            self.config.command,
            self.config.env
        )
        if await new_client.connect():
            self._connections.append(new_client)
            await self._available.put(new_client)
            print(f"🔄 MCP 连接已替换 ({self.config.name})")
        else:
            await new_client.disconnect()
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self._state == ConnectionState.CONNECTED:
            await asyncio.sleep(self.config.health_check_interval)
            
            if self._state != ConnectionState.CONNECTED:
                break
            
            await self._perform_health_check()
    
    async def _perform_health_check(self):
        """执行健康检查"""
        self._last_health_check = time.time()
        
        # 检查所有空闲连接
        unhealthy_clients = []
        temp_clients = []
        
        # 取出所有可用连接进行检查
        while not self._available.empty():
            try:
                client = self._available.get_nowait()
                temp_clients.append(client)
            except asyncio.QueueEmpty:
                break
        
        for client in temp_clients:
            if await self._is_healthy(client):
                await self._available.put(client)
            else:
                unhealthy_clients.append(client)
        
        # 替换不健康的连接
        for client in unhealthy_clients:
            print(f"⚠️ MCP 连接不健康，准备替换 ({self.config.name})")
            await self._replace_connection(client)
            self._consecutive_failures += 1
        
        if not unhealthy_clients:
            self._consecutive_failures = 0
        
        # 连续失败过多，标记为不健康
        if self._consecutive_failures >= self.config.max_retries:
            self._state = ConnectionState.UNHEALTHY
            print(f"❌ MCP 服务器标记为不健康 ({self.config.name})")
            # 尝试重连
            asyncio.create_task(self._reconnect())
    
    async def _reconnect(self):
        """重连逻辑"""
        if self._state == ConnectionState.RECONNECTING:
            return
        
        self._state = ConnectionState.RECONNECTING
        delay = self.config.retry_delay
        
        for attempt in range(self.config.max_retries):
            print(f"🔄 MCP 尝试重连 ({self.config.name}) 第 {attempt + 1} 次")
            await asyncio.sleep(delay)
            
            success = await self.initialize()
            if success:
                print(f"✅ MCP 重连成功 ({self.config.name})")
                return
            
            delay *= self.config.retry_backoff
        
        print(f"❌ MCP 重连失败，已达最大重试次数 ({self.config.name})")
        self._state = ConnectionState.UNHEALTHY
    
    async def shutdown(self):
        """关闭连接池"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # 关闭所有连接
        for client in self._connections:
            await client.disconnect()
        
        self._connections.clear()
        self._in_use.clear()
        
        while not self._available.empty():
            try:
                self._available.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        self._state = ConnectionState.DISCONNECTED


def import_os_environ():
    import os
    return dict(os.environ)


class MCPPlugin(AgentPlugin):
    """MCP 支持插件 - 管理多个 MCP 服务器连接池（健康检查、自动重连）"""
    
    name = "mcp"
    priority = 25  # 在 ToolPlugin 之前加载，以便注册工具
    
    def __init__(self, config=None):
        super().__init__(config)
        self._pools: Dict[str, MCPConnectionPool] = {}
        self._mcp_config: List[Dict[str, Any]] = []
    
    def _initialize(self) -> None:
        # 从配置读取 MCP 服务器列表
        self._mcp_config = getattr(self.config, 'mcp_servers', [])
        
        if self._mcp_config:
            # 创建异步任务连接服务器
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
        """添加并连接 MCP 服务器（使用连接池）
        
        Args:
            server_config: {
                "name": "github",           # 服务器名称
                "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_TOKEN": "xxx"},  # 可选环境变量
                # 连接池配置（可选）
                "pool_size": 2,
                "health_check_interval": 30,
                "max_retries": 3,
                "retry_delay": 5.0,
            }
        """
        name = server_config.get("name")
        command = server_config.get("command")
        env = server_config.get("env", {})
        
        if not name or not command:
            print(f"❌ MCP 配置无效: {server_config}")
            return False
        
        if name in self._pools:
            print(f"⚠️ MCP 服务器已存在: {name}")
            return True
        
        # 创建服务器配置
        pool_config = MCPServerConfig(
            name=name,
            command=command,
            env=server_config.get("env", {}),
            pool_size=server_config.get("pool_size", 1),
            health_check_interval=server_config.get("health_check_interval", 30),
            health_check_timeout=server_config.get("health_check_timeout", 10),
            max_retries=server_config.get("max_retries", 3),
            retry_delay=server_config.get("retry_delay", 5.0),
            retry_backoff=server_config.get("retry_backoff", 2.0),
            request_timeout=server_config.get("request_timeout", 30.0),
            connect_timeout=server_config.get("connect_timeout", 10.0),
        )
        
        # 创建连接池
        pool = MCPConnectionPool(pool_config)
        success = await pool.initialize()
        
        if success:
            self._pools[name] = pool
            # 注册工具（从第一个连接获取工具列表）
            # 获取一个连接来列出工具
            client = await pool.acquire()
            if client:
                await self._register_tools(client)
                await pool.release(client)
                print(f"✅ MCP 服务器已连接: {name} ({pool.config.pool_size} 连接池)")
                return True
        
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
            
            # 创建 MCP 工具（使用连接池感知的包装器）
            mcp_tool = PooledMCPTool(
                name=tool_name,
                description=tool_def.get("description", ""),
                parameters=parameters,
                pool=self._pools[client.server_name.split("-")[0]],  # 从池获取连接
                original_name=tool_def["name"]
            )
            
            self.context.tool_registry.register_tool(mcp_tool)
    
    async def remove_server(self, name: str):
        """移除 MCP 服务器"""
        if name in self._pools:
            await self._pools[name].shutdown()
            del self._pools[name]
            
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
        return list(self._pools.keys())
    
    def get_pool_status(self) -> Dict[str, Any]:
        """获取所有连接池状态"""
        return {
            name: {
                "state": pool.state.value,
                "pool_size": pool.config.pool_size,
                "available": pool._available.qsize(),
                "in_use": len(pool._in_use),
                "total_connections": len(pool._connections),
                "consecutive_failures": pool._consecutive_failures,
                "last_health_check": pool._last_health_check,
            }
            for name, pool in self._pools.items()
        }
    
    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """调用 MCP 工具（带服务器前缀）"""
        # 解析服务器名称
        if "__" not in tool_name:
            return f"❌ 无效的 MCP 工具名格式: {tool_name} (应为 server__tool)"
        
        server_name, original_name = tool_name.split("__", 1)
        if server_name in self._pools:
            pool = self._pools[server_name]
            client = await pool.acquire()
            if client:
                try:
                    return await client.call_tool(original_name, arguments)
                finally:
                    await pool.release(client)
        return f"❌ MCP 服务器未找到: {server_name}"
    
    async def teardown(self):
        """清理所有连接"""
        for pool in self._pools.values():
            await pool.shutdown()
        self._pools.clear()


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