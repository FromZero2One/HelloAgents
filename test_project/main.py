#!/usr/bin/env python
"""HelloAgents 项目入口"""

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
            user_input = input("\n> ")
            if user_input.lower() in ("exit", "quit"):
                break
            result = agent.run(user_input)
            print(f"\n{result}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    main()
