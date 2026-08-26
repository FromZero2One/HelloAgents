# AGENTS.md - HelloAgents

## Project Overview
**HelloAgents** - 生产级多智能体框架 (Python 3.10+)
- Package name: `hello-agents` (pip installable)
- License: CC BY-NC-SA 4.0 (non-commercial only)
- Version: v1.0.0-dev (main branch)
- Two branches: `main` (dev, v1.0.0) and `learn_version` (stable tutorial version)

## Developer Commands

### Install & Setup
```bash
pip install -e .                    # Development install
pip install hello-agents            # From PyPI
cp .env.example .env                # Configure LLM credentials
```

## Testing
```bash
pytest                              # All tests (217 tests, 20 test files)
pytest tests/test_all_agents.py     # Single test file (integration tests)
pytest -v --tb=short                # Verbose with short traceback
```

### Code Quality
```bash
black hello_agents tests            # Format
isort hello_agents tests            # Import sorting
mypy hello_agents                   # Type checking
```

### Run Examples
```bash
python examples/subagent_demo.py
python examples/skills_demo.py
python examples/async_agent_demo.py
python examples/fastapi_sse_server.py  # SSE server
```

## Architecture Notes

### Package Structure
```
hello_agents/
├── core/           # LLM基类、适配器、Agent基类、会话存储、生命周期、流式输出、AgentGraph、插件系统
├── agents/         # 4种Agent实现: SimpleAgent, ReActAgent, ReflectionAgent, PlanAndSolveAgent + AgentMVP + Factory
├── tools/          # 工具系统: 注册表、响应协议、熔断器、工具过滤、内置工具、MCP连接池
├── context/        # 上下文工程: HistoryManager, TokenCounter, Truncator, ContextBuilder
├── observability/  # TraceLogger 追踪系统、OpenTelemetry 指标/插桩
├── skills/         # SkillLoader 技能系统
├── cli.py          # CLI 脚手架 (init/run/benchmark/version)
```

### Key Entry Points
- `hello_agents/__init__.py` - Public exports
- `hello_agents/core/llm.py` - `HelloAgentsLLM` (auto-detects provider from base_url)
- `hello_agents/agents/factory.py` - `AgentFactory` for creating agents
- `hello_agents/tools/registry.py` - `ToolRegistry` for tool management
- `hello_agents/core/graph.py` - `AgentGraph` for agent orchestration
- `hello_agents/observability/plugins_otel.py` - `trace_metric` decorator for OTel
- `hello_agents/cli.py` - CLI entry point

### LLM Provider Auto-Detection
Framework auto-selects adapter based on `LLM_BASE_URL`:
- OpenAI compatible (default): OpenAI, DeepSeek, Qwen, Kimi, GLM, vLLM, Ollama
- Anthropic: `anthropic.com` in URL
- Gemini: `googleapis.com` or `generativelanguage` in URL

### 16 Core Capabilities
1. ToolResponse 统一返回格式
2. HistoryManager/TokenCounter/Truncator/ContextBuilder (上下文工程)
3. SessionStore (会话持久化)
4. TaskTool + ToolFilter (子代理机制)
5. 乐观锁 (文件编辑并发控制)
6. CircuitBreaker (熔断器)
7. Skills 知识外化
8. TodoWrite 进度管理
9. DevLog 决策日志
10. SSE 流式输出
11. 异步生命周期
12. TraceLogger 可观测性
13. 四种日志范式
14. Function Calling 架构重构
15. LLM/Agent 基类重构
16. 自定义工具扩展 (3种方式)
17. Pydantic v2 序列化
18. AgentGraph 编排 (循环检测/可视化导出)
19. OpenTelemetry 指标/自动插桩
20. MCP 连接池 (连接复用/熔断/工具发现缓存)
21. CLI 脚手架 (init/run/benchmark/version)

## Testing Quirks
- Tests require `.env` with valid LLM credentials for integration tests
- `test_all_agents.py` runs all agent types
- Some tests use real API calls (network required)

## Config Files
- `pyproject.toml` - Build config, pytest, black, isort, mypy settings
- `.env` - LLM credentials (not committed)
- `.gitignore` - Excludes `__pycache__`, `.env`, `*.egg-info`, `dist/`, `build/`

## Important Constraints
- **Non-commercial license** - Cannot use commercially without permission
- Python 3.10+ required (uses modern typing features)
- Pydantic v2 only (v1 not supported)
- Optional deps: `gemini` (google-genai), `anthropic` (anthropic SDK)