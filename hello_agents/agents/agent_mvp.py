import json
import os
from openai import OpenAI

# ========== 配置 ==========
# 使用本地 Ollama 模型（兼容 OpenAI API 格式）
client = OpenAI(
    api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
)
MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
MAX_STEPS = 5


# ========== 定义工具 ==========
def get_weather(city: str) -> str:
    """获取指定城市的天气（模拟）"""
    # MVP 阶段先用模拟数据，后续可替换为真实 API
    return f"{city}：晴，25°C，湿度 60%"


def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


# 工具注册表
TOOLS = {
    "get_weather": {
        "fn": get_weather,
        "description": "获取城市天气，参数: city (字符串)"
    },
    "calculate": {
        "fn": calculate,
        "description": "计算数学表达式，参数: expression (字符串，如 '3+5*2')"
    }
}

# 生成 OpenAI 格式的工具描述
TOOL_SCHEMAS = [{
    "type": "function",
    "function": {
        "name": name,
        "description": info["description"],
        "parameters": {"type": "object", "properties": {}}
    }
} for name, info in TOOLS.items()]

# ========== 系统提示词 ==========
SYSTEM_PROMPT = """你是一个智能助手，可以使用工具来帮助用户。
当需要调用工具时，请使用函数调用格式。
如果你已经有了答案，直接回复用户。"""


# ========== 主循环 ==========
def run_agent(user_input: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    for step in range(MAX_STEPS):
        print(f"\n{'=' * 40} 第 {step + 1} 轮 {'=' * 40}")

        # 1. 调用模型
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto"
        )

        msg = response.choices[0].message
        messages.append(msg.model_dump())

        # 2. 检查是否需要调用工具
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"🔧 调用工具: {tool_name}({args})")

                # 执行工具
                if tool_name in TOOLS:
                    result = TOOLS[tool_name]["fn"](**args)
                else:
                    result = f"未知工具: {tool_name}"

                print(f"📊 工具返回: {result}")

                # 将工具结果加入对话
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            # 3. 没有工具调用，输出最终答案
            print(f"\n✅ 最终答案:\n{msg.content}")
            return msg.content

    print(f"\n⚠️ 达到最大步数 {MAX_STEPS}，停止。")
    return None


# ========== 运行测试 ==========
if __name__ == "__main__":
    print("🤖 Agent MVP 已启动！")
    print("支持工具: get_weather, calculate")
    while True:
        user_input = input("\n💬 你: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        run_agent(user_input)