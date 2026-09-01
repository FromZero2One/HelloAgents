"""MCPPlugin - Model Context Protocol 支持插件

职责：
- 连接到 MCP 服务器（支持连接池、健康检查、自动重连）
- 发现并注册 MCP 工具
- 代理工具调用到 MCP 服务器
- 支持多个 MCP 服务器连接
"""

from typing import List, Dict, Any, Optional, Callable, Set
from .plugins import AgentPlugin, PluginContext
import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from ..tools.circuit_breaker import CircuitBreaker
from ..tools.errors import ToolErrorCode


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
    """MCP 连接池 - 管理单个服务器的多个连接（含熔断器保护）"""
    
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
        # 熔断器 - 保护整个连接池
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=config.max_retries,
            recovery_timeout=int(config.retry_delay * config.retry_backoff ** (config.max_retries - 1)),
            name=f"mcp_pool_{config.name}"
        )
    
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
        """获取可用连接（带熔断器保护）"""
        # 检查熔断器
        if self._circuit_breaker.is_open():
            status = self._circuit_breaker.get_status()
            print(f"⚠️ MCP 连接池熔断开启 ({self.config.name}), {status['recover_in_seconds']:.0f}秒后恢复")
            return None
        
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
    """MCP 支持插件 - 管理多个 MCP 服务器连接池（健康检查、自动重连、工具发现缓存、熔断器）"""
    
    name = "mcp"
    priority = 25  # 在 ToolPlugin 之前加载，以便注册工具
    
    def __init__(self, config=None):
        super().__init__(config)
        self._pools: Dict[str, MCPConnectionPool] = {}
        self._mcp_config: List[Dict[str, Any]] = []
        # 工具发现缓存
        self._tool_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl: int = 300  # 5分钟缓存TTL
    
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
    
    # ==================== 工具发现缓存 ====================
    
    def _is_cache_valid(self, server_name: str) -> bool:
        """检查缓存是否有效"""
        if server_name not in self._tool_cache:
            return False
        age = time.time() - self._cache_timestamps.get(server_name, 0)
        return age < self._cache_ttl
    
    def get_tools_cached(self, server_name: str) -> List[Dict[str, Any]]:
        """获取工具列表（优先使用缓存）"""
        if self._is_cache_valid(server_name):
            return self._tool_cache[server_name]
        return []
    
    async def refresh_tool_cache(self, server_name: str) -> List[Dict[str, Any]]:
        """刷新工具缓存（从服务器获取最新工具列表）"""
        if server_name not in self._pools:
            return []
        
        pool = self._pools[server_name]
        client = await pool.acquire()
        if not client:
            return self.get_tools_cached(server_name)
        
        try:
            tools = await client.list_tools()
            self._tool_cache[server_name] = tools
            self._cache_timestamps[server_name] = time.time()
            return tools
        except Exception as e:
            print(f"⚠️ 刷新工具缓存失败 ({server_name}): {e}")
            return self.get_tools_cached(server_name)
        finally:
            await pool.release(client)
    
    async def refresh_all_caches(self):
        """刷新所有服务器的工具缓存"""
        for server_name in self._pools:
            await self.refresh_tool_cache(server_name)
    
    def set_cache_ttl(self, ttl: int):
        """设置缓存TTL（秒）"""
        self._cache_ttl = max(60, ttl)  # 最小60秒
    
    def clear_cache(self, server_name: Optional[str] = None):
        """清除缓存"""
        if server_name:
            self._tool_cache.pop(server_name, None)
            self._cache_timestamps.pop(server_name, None)
        else:
            self._tool_cache.clear()
            self._cache_timestamps.clear()
    
    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """调用 MCP 工具（带服务器前缀，熔断器保护）"""
        # 解析服务器名称
        if "__" not in tool_name:
            return f"❌ 无效的 MCP 工具名格式: {tool_name} (应为 server__tool)"
        
        server_name, original_name = tool_name.split("__", 1)
        if server_name in self._pools:
            pool = self._pools[server_name]
            # 检查熔断器
            if pool._circuit_breaker.is_open():
                status = pool._circuit_breaker.get_status()
                return f"❌ MCP 服务器熔断开启 ({server_name}), {status['recover_in_seconds']:.0f}秒后恢复"
            
            client = await pool.acquire()
            if client:
                try:
                    result = await client.call_tool(original_name, arguments)
                    # 记录成功
                    pool._circuit_breaker.record_result(True)
                    return result
                except Exception as e:
                    # 记录失败
                    pool._circuit_breaker.record_result(False)
                    return f"❌ MCP 工具调用失败: {e}"
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


class PooledMCPTool(MCPTool):
    """MCP 工具包装器（连接池版）- 每次调用从连接池获取连接"""
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: List,
        pool: MCPConnectionPool,
        original_name: str
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.pool = pool
        self.original_name = original_name
        self._expandable = False
        self.mcp_client = None  # 不使用固定客户端，改用连接池
    
    async def arun(self, arguments: Dict[str, Any]) -> str:
        """异步运行（从连接池获取连接）"""
        client = await self.pool.acquire()
        if not client:
            return f"❌ MCP 连接池无可用连接 ({self.pool.config.name})"
        try:
            return await client.call_tool(self.original_name, arguments)
        except Exception as e:
            return f"❌ MCP 工具调用失败: {e}"
        finally:
            await self.pool.release(client)
    
    def run(self, arguments: Dict[str, Any]) -> str:
        """同步运行（不推荐，建议使用异步）"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.arun(arguments))