"""HelloAgents CLI

提供命令行工具：init, run, benchmark
"""

import sys
import argparse
from pathlib import Path
from typing import Optional


def cmd_init(args: argparse.Namespace) -> int:
    """初始化新项目"""
    project_dir = Path(args.path).resolve()
    
    if project_dir.exists() and any(project_dir.iterdir()):
        print(f"[ERROR] 目录 {project_dir} 不为空")
        return 1
    
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建基础目录结构
    (project_dir / "agents").mkdir(exist_ok=True)
    (project_dir / "tools").mkdir(exist_ok=True)
    (project_dir / "skills").mkdir(exist_ok=True)
    (project_dir / "config").mkdir(exist_ok=True)
    
    # 创建 .env.example
    env_example = project_dir / ".env.example"
    env_example.write_text("""# LLM 配置
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# 可选：其他提供商
# LLM_API_KEY=your-key
# LLM_BASE_URL=https://api.deepseek.com
# LLM_MODEL=deepseek-chat
""", encoding="utf-8")
    
    # 创建主 Agent 文件
    main_py = project_dir / "main.py"
    main_py.write_text("""#!/usr/bin/env python
\"\"\"HelloAgents 项目入口\"\"\"

import os
from hello_agents import HelloAgentsLLM, SimpleAgent, ToolRegistry
from hello_agents.tools.builtin.calculator import CalculatorTool


def main():
    # 从环境变量加载配置
    llm = HelloAgentsLLM()
    
    # 注册工具
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    
    # 创建 Agent
    agent = SimpleAgent(llm=llm, tools=registry)
    
    # 运行
    print("HelloAgents 启动成功！输入 'exit' 退出。")
    while True:
        try:
            user_input = input("\\n> ")
            if user_input.lower() in ("exit", "quit"):
                break
            result = agent.run(user_input)
            print(f"\\n{result}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    main()
""", encoding="utf-8")
    
    # 创建 requirements.txt
    req = project_dir / "requirements.txt"
    req.write_text("hello-agents>=1.0.0\n", encoding="utf-8")
    
    # 创建 README.md
    readme = project_dir / "README.md"
    readme.write_text(f"""# {project_dir.name}

HelloAgents 项目

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 运行
python main.py
```

## 目录结构

```
.
├── agents/      # 自定义 Agent
├── tools/       # 自定义工具
├── skills/      # 技能文件
├── config/      # 配置文件
├── main.py      # 入口文件
└── .env         # 环境变量（不提交）
```
""", encoding="utf-8")
    
    print(f"[OK] 项目已创建: {project_dir}")
    print(f"   目录结构: agents/, tools/, skills/, config/")
    print(f"   入口文件: main.py")
    print(f"   配置模板: .env.example")
    print(f"\n下一步:")
    print(f"   cd {project_dir.name}")
    print(f"   cp .env.example .env  # 编辑填入 API Key")
    print(f"   pip install -r requirements.txt")
    print(f"   python main.py")

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """运行 Agent"""
    # 加载环境变量
    try:
        from dotenv import load_dotenv
        load_dotenv(args.env_file)
    except ImportError:
        pass
    
    # 导入用户的 main 模块
    sys.path.insert(0, str(Path(args.file).parent))
    
    try:
        if args.file.endswith(".py"):
            module_name = Path(args.file).stem
            import importlib.util
            spec = importlib.util.spec_from_file_location(module_name, args.file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, "main"):
                module.main()
            else:
                print("[ERROR] 文件中未找到 main() 函数")
                return 1
        else:
            print("[ERROR] 仅支持 .py 文件")
            return 1
    except Exception as e:
        print(f"[ERROR] 运行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """运行性能基准测试"""
    from hello_agents.tools.registry import ToolRegistry
    from hello_agents.tools.builtin.calculator import CalculatorTool
    import time
    import asyncio
    
    print("HelloAgents 性能基准测试")
    print("=" * 50)
    
    # 测试工具注册
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    
    # 同步调用测试
    print("\n[1/3] 同步工具调用测试")
    test_cases = [
        "2 + 3",
        "10 * 20",
        "sqrt(144) + 5",
        "sin(3.14159/2)",
        "max(1, 2, 3, 4, 5)",
    ]
    
    start = time.perf_counter()
    for expr in test_cases * 20:  # 100 次调用
        registry.execute_tool("python_calculator", expr)
    sync_time = time.perf_counter() - start
    print(f"   100 次调用耗时: {sync_time*1000:.1f}ms")
    print(f"   吞吐量: {100/sync_time:.1f} ops/s")
    
    # 并发调用测试
    print("\n[2/3] 并发工具调用测试")
    import concurrent.futures
    
    def run_batch(expressions):
        reg = ToolRegistry()
        reg.register_tool(CalculatorTool())
        for expr in expressions:
            reg.execute_tool("python_calculator", expr)
    
    batch_size = 20
    num_workers = 4
    expressions = test_cases * 5  # 100 个表达式
    
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(run_batch, expressions[i:i+batch_size])
            for i in range(0, len(expressions), batch_size)
        ]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    parallel_time = time.perf_counter() - start
    print(f"   100 次调用耗时: {parallel_time*1000:.1f}ms (workers={num_workers})")
    print(f"   吞吐量: {100/parallel_time:.1f} ops/s")
    print(f"   加速比: {sync_time/parallel_time:.2f}x")
    
    # Agent 创建测试
    print("\n[3/3] Agent 创建测试")
    try:
        from hello_agents import HelloAgentsLLM, SimpleAgent
        start = time.perf_counter()
        for i in range(10):
            try:
                llm = HelloAgentsLLM()
                agent = SimpleAgent(name=f"test_{i}", llm=llm, tool_registry=registry)
            except Exception:
                # 如果LLM未配置，使用mock
                from unittest.mock import MagicMock
                llm = MagicMock()
                agent = SimpleAgent(name=f"test_{i}", llm=llm, tool_registry=registry)
        agent_time = time.perf_counter() - start
        print(f"   创建 10 个 Agent 耗时: {agent_time*1000:.1f}ms")
    except Exception as e:
        print(f"   [SKIP] Agent 创建测试跳过: {e}")

    print("\n[OK] 基准测试完成")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """显示版本信息"""
    from .version import __version__
    print(f"HelloAgents v{__version__}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="hello-agents",
        description="HelloAgents - 灵活、可扩展的多智能体框架"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # init 命令
    init_parser = subparsers.add_parser("init", help="初始化新项目")
    init_parser.add_argument("path", nargs="?", default=".", help="项目路径 (默认: 当前目录)")
    init_parser.set_defaults(func=cmd_init)
    
    # run 命令
    run_parser = subparsers.add_parser("run", help="运行 Agent")
    run_parser.add_argument("file", help="入口文件路径 (如 main.py)")
    run_parser.add_argument("--env", dest="env_file", default=".env", help="环境变量文件")
    run_parser.set_defaults(func=cmd_run)
    
    # benchmark 命令
    bench_parser = subparsers.add_parser("benchmark", help="运行性能基准测试")
    bench_parser.set_defaults(func=cmd_benchmark)
    
    # version 命令
    version_parser = subparsers.add_parser("version", help="显示版本信息")
    version_parser.set_defaults(func=cmd_version)
    
    args = parser.parse_args()
    
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())