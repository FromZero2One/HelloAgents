# 优化路线图 (Optimization Roadmap)

> 本文档梳理 HelloAgents 项目当前需要优化的问题与改进方向，便于后续按优先级修复。
> 整理时间：2026-08-27 ｜ 适用范围：v1.0.0-dev

---

## 优先级总览

| 优先级 | 数量 | 类别 |
|--------|------|------|
| 🔴 P0（阻塞/必修） | 2 | MCP 插件、CLI 脚手架 |
| 🟡 P1（功能缺陷） | 3 | 智能摘要、ReAct 历史、流式 ReAct 双重调用 |
| 🟢 P2（代码质量） | 4 | 类型注解、依赖治理、构建/配置噪音、调试日志 |
| 🔵 P3（增强建议） | 4 | 上下文工程回填、Test Mock 分层、可视化输出、文档同步 |

---

## 🔴 P0：阻塞级问题（必修）

### 1. MCP 插件整体不可运行

**位置**：`hello_agents/core/plugins_mcp.py`（595 行）

**问题清单**（全部已确认到行号）：
- `MCPClient` 类在**整个仓库内无任何定义**（`rg "class MCPClient"` 零匹配），却被多处引用：
  - `MCPConnectionPool.__init__` L55 声明 `List[MCPClient]`
  - `MCPConnectionPool.initialize` L84 调用 `MCPClient(...)`
  - `MCPConnectionPool.acquire/release` L104/130 返回/接收 `MCPClient`
  - `MCPConnectionPool._replace_connection` L160 再次 `MCPClient(...)`
  - `MCPPlugin._register_tools` L368/394 接收 `MCPClient` 形参
- `PooledMCPTool` 类同样**完全缺失**（仅 L394 引用）
- `Set` 未导入：`from typing import List, Dict, Any, Optional, Callable`（L10）缺 `Set`，但 L57 用 `Set[MCPClient]`
- `CircuitBreaker` 调用签名错误（与 `tools/circuit_breaker.py` 实际签名不匹配）：
  - L64-68 构造传 `name=...`（实际签名仅 `failure_threshold/recovery_timeout/enabled`）
  - L107 `breaker.is_open()` 无参（实现要求 `tool_name`）
  - L108 `breaker.get_status()` 无参（实现要求 `tool_name`）
  - L511/515 `record_result(True/False)`（实现要求 `(tool_name, response)`）

**影响**：只要 `Config.mcp_servers` 非空启用 MCP 插件，必然 `NameError` 或 `TypeError`。当前因默认配置该字段为空，暂未暴露。

**修复建议**：
1. 引入 `MCPClient` 实现（封装 `mcp` SDK 的 `stdio_client`/`SSE 客户端`，需新增 `mcp` 为可选依赖）
2. 引入 `PooledMCPTool`（继承 `Tool`，`run`/`arun` 走连接池）
3. 修正 `Set` 导入
4. 修正所有 CircuitBreaker 调用以匹配实际签名
5. 在 `pyproject.toml` 将 `mcp` 加入 `[project.optional-dependencies].mcp`
6. 增加 `examples/mcp_demo.py` 与对应 `tests/test_mcp_plugin.py`

---

### 2. CLI 脚手架生成的代码用了过时的 API

**位置**：`hello_agents/cli.py:cmd_init`（L12-）

**问题**：
- 生成器模板调用 `registry.register(CalculatorTool())`（L57 风格）
- 而实际 API 是 `registry.register_tool()`（`tools/registry.py:30`）
- `SimpleAgent(llm=llm, tools=registry)` 与真实构造签名不一致（真实是 `tool_registry=`）

**影响**：按 `hello-agents init` 生成的项目第一次运行就会报错，**新手入门完全断链**。

**修复建议**：
1. 重写 `cmd_init` 中 main.py 模板，使用 `register_tool()` 与正确 kwargs
2. 增加 `hello_agents init` 后的 smoke test（生成后立即 import 检查）
3. 在 docs 中标注 init 模板的最低支持版本

---

## 🟡 P1：功能缺陷

### 3. 智能摘要路径 TypeError

**位置**：`hello_agents/core/agent.py:534`

**问题**：
```python
summary_llm = HelloAgentsLLM(provider=provider, ...)
```
但 `HelloAgentsLLM.__init__`（`core/llm.py:28-37`）**没有 `provider` 参数**。

**影响**：启用 `enable_smart_compression=True` 时必然 TypeError。当前默认 False，所以不触发；但文档/示例里凡提到智能摘要路径都是不可用的。

**修复建议**：
1. 要么在 `HelloAgentsLLM` 真正加 `provider` 参数（首选，可让用户绕过自动检测）
2. 要么删除 `_get_summary_llm` 中的 `provider` 参数，改用环境变量/Config
3. 添加单元测试覆盖 smart summary 路径，避免再次回归

---

### 4. ReActAgent 不携带历史消息

**位置**：`hello_agents/agents/react_agent.py:389`（`_build_messages`）

**问题**：
- `_build_messages()` 仅组装 `system + 当前 user`，不读 `self._history`
- 对比 `SimpleAgent._build_messages()`（`simple_agent.py:270`）**包含 `self._history`**
- 行为不一致：同一项目下 ReAct 多轮对话失忆

**影响**：用户体验割裂，ReAct 范式几乎无法做交互式多轮任务。

**修复建议**：
1. 让 ReAct 复用 `SimpleAgent` 的消息构建逻辑或提取共同基类 `BaseFunctionCallAgent._build_messages()`
2. 加 test 断言 history 注入正确
3. 在 `react_agent_guide.md`（待补）说明 history 行为

---

### 5. 流式 ReAct 双重 LLM 调用

**位置**：`hello_agents/agents/react_agent.py:977-986`（`arun_stream`）

**问题**：
- 先整段流式拿到 assistant 文本
- 再发起一次**非流式** `invoke_with_tools` 获取 tool_calls
- 注释 L978 自承「简化处理」
- Token 成本翻倍，且两次输出可能不一致

**影响**：用户用 `arun_stream` 跑带工具的 ReAct 时成本和延迟都翻倍，且偶尔会出现"流式输出的文本"与"实际决策的工具"对不上的诡异现象。

**修复建议**：
1. 改为单次调用：用支持 streaming tool_calls 的 OpenAI 客户端（`stream=True` 时也能返回 `delta.tool_calls`），边流式收文本边累积 tool_calls 块
2. 或在 `LLMResponse` 流式事件中加 `ToolCallDelta` 类型
3. 短期方案：明确文档标注此为已知简化

---

## 🟢 P2：代码质量

### 6. 类型注解严重缺失

**问题**：
- `pyproject.toml` 配置 mypy `disallow_untyped_defs=true`，但源码 90% 函数无类型注解
- CI 实际跑不起来或全文件被 skip

**修复建议**：
1. 短期：放宽 mypy 配置（`check_untyped_defs=false`、`ignore_missing_imports=true`），或暂时移除 strict 字段
2. 中期：分模块补齐注解，先从公共 API（`Agent`/`HelloAgentsLLM`/`Tool`/`ToolRegistry`）入手
3. 加 `mypy hello_agents/core/llm.py --follow-imports=silent` 作为渐进门槛

---

### 7. 依赖与构建治理

**问题**：
- 双构建痕迹：`setup.py`、`pyproject.toml`、`uv.lock` 三者并存
- `networkx` 声明为依赖但 `core/graph.py` 用手写 Tarjan，没真正用到 networkx
- OTel 相关包（`opentelemetry-api/sdk/exporter`）未列入 `[project.optional-dependencies]`，需手动安装
- `.env.example` 中残留 Qdrant/Neo4j/embedding 变量，对应实现已删除（ContextBuilder docstring 已注明 RAG 移除）
- 仓库根散落临时文件 `fix_indent.py`、`benchmark_results.json` 等

**修复建议**：
1. 删除 `setup.py`，统一用 `pyproject.toml`
2. 删除 `uv.lock`（若非团队统一用 uv）或保留并固化 uv 工作流
3. 移除 `networkx` 依赖或真正用起来（拓扑可视化、布局算法）
4. 在 `[project.optional-dependencies]` 加 `opentelemetry` group
5. 清理 `.env.example` 残留变量
6. 把临时脚本移到 `scripts/` 或加进 `.gitignore`

---

### 8. 框架级 print 噪音

**问题**：
- 大量 `print("⚠️...")` / `print("❌...")` 直接打到 stdout（`plugins_mcp.py` 尤其严重）
- 没有走 `logging` 模块，无法分级 / 静默 / 重定向

**修复建议**：
1. 引入 `hello_agents/core/logger.py`，统一 `get_logger(__name__)`
2. 用 `logger.warning/debug/error` 替换所有 print
3. 加 `Config.log_level` 字段控制全局日志级别
4. 默认行为保持静默，避免污染用户 stdout

---

### 9. 编码与文件噪声

**问题**：
- 部分源文件存在 GBK/UTF-8 编码痕迹（`version.py` 等）
- 仓库提交了 `.env`（虽然 `.gitignore` 已声明）

**修复建议**：
1. 全仓库统一 UTF-8 with BOM 检查 + `.editorconfig`
2. 在 CI 加 `git check-ignore .env` 防止再次误提交
3. 清理版本文件中残留的乱码注释

---

## 🔵 P3：增强建议

### 10. 上下文工程回填

**位置**：`hello_agents/context/builder.py:52`（`ContextBuilder`）

**问题**：docstring 明确声明"此类暂时不可用"，`MemoryTool/RAGTool` 已删除。

**建议**：
- 要么删除 `ContextBuilder`（避免误导）
- 要么补一个真正能用的轻量实现（如基于 token 预算的对话摘要压缩）
- 在 `ContextBuilder` 顶部加 `DEPRECATED` 警告或整个 docstring 重写

---

### 11. 测试 Mock 分层

**问题**：
- 217 个测试绝大多数依赖真实 LLM 凭据
- 没有 `conftest.py`、没有 marker 分层，无法一键跳过联网测试
- CI 无法在没有 secret 的 PR 上跑

**建议**：
1. 增加 `pytest.ini` / `pyproject.toml [tool.pytest.ini_options]` 的 markers：
   ```toml
   markers = [
     "integration: 需要真实 LLM API 的集成测试",
     "slow: 慢测试（>5s）",
     "network: 需要网络",
   ]
   ```
2. 拆分集成 vs 单元：在 CI 默认 `pytest -m "not integration"`，nightly job 跑全量
3. 对 `HelloAgentsLLM` 封装 `MockLLMAdapter`（已部分存在），统一 mock 接口

---

### 12. AgentGraph 可视化输出

**问题**：
- 仓库根的 `test_graph.dot/.json/.mmd` 显然是开发过程产物
- `to_graphviz`/`to_mermaid` 没有标准化输出目录

**建议**：
1. 把测试输出统一到 `tests/outputs/` 并加进 `.gitignore`
2. 增加 `AgentGraph.export()` 默认输出到 `./graphs/<name>.<format>`

---

### 13. 文档同步

**问题**：
- `docs/` 16 篇与 `examples/` 17 个 demo 一一对应
- 但缺：ReAct 行为说明（含多轮历史）、智能摘要路径、MCP 集成指南

**建议**：
- 新增 `docs/react-agent-behavior.md`（含 history 行为说明）
- 新增 `docs/mcp-integration.md`（修复完 P0-1 后）
- 新增 `docs/smart-summary.md`（修复完 P1-3 后）

---

## 修复顺序建议

```
第 1 周：P0-1 (MCP) + P0-2 (CLI 脚手架)    → 阻塞项清零
第 2 周：P1-3 (智能摘要) + P1-4 (ReAct 历史) → 一致性修复
第 3 周：P1-5 (流式 ReAct) + P2-8 (logger)  → 体验提升
第 4 周：P2-6 (类型注解) + P2-7 (依赖治理)  → 长期健康度
持续：   P3 类增量改进
```

---

## 相关参考

- 代码库调研报告（本次梳理）：见会话历史
- `hello_agents/core/plugins.py:206` - `create_default_plugins`（插件清单）
- `hello_agents/agents/factory.py:15` - `create_agent`（Agent 工厂）
- `hello_agents/core/llm_adapters.py:860` - `create_adapter`（provider 检测）
- `hello_agents/core/agent.py:534` - `_get_summary_llm`（P1-3 现场）
- `hello_agents/agents/react_agent.py:389` - `_build_messages`（P1-4 现场）