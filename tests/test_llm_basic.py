"""HelloAgentsLLM 基础功能测试

覆盖范围：
  1. invoke            - 同步非流式
  2. stream_invoke     - 同步流式
  3. invoke_with_tools - Function Calling
  4. ainvoke           - 异步非流式
  5. astream_invoke    - 异步流式

环境: Ollama 本地服务 (localhost:11434) + qwen3.5:latest（推理模型）
说明: qwen3.5 是推理模型，会先输出大量 thinking 再给答案。
      所有测试统一 max_tokens=500 限制 + 极简 prompt，控制单次 5-15s。
"""

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

# 把项目根加入 path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 清除代理环境变量（本地 Ollama 不需要代理）
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
    os.environ.pop(_k, None)

# 手动加载 .env
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from hello_agents import HelloAgentsLLM

# 测试用 prompt —— qwen3.5 是推理模型，会先 thinking
# 不加 system prompt（容易触发大段思考），max_tokens 给足
SIMPLE_PROMPT = [{"role": "user", "content": "1+1=?"}]
SYSTEM_PROMPT = [{"role": "user", "content": "说hi"}]
TOOL_PROMPT = [{"role": "user", "content": "北京天气？"}]
ASYNC_PROMPT = [{"role": "user", "content": "1+1=?"}]
ASTREAM_PROMPT = [{"role": "user", "content": "1+2=?"}]

MAX_TOKENS = 32 * 1024  # 推理模型 thinking 占大头，需要给足预算


def banner(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_0_init():
    from openai import OpenAI
    """
    最基本的大模型调用
    """

    # 创建客户端
    _client = OpenAI(
        api_key='ollama',
        base_url='http://127.0.0.1:11434/v1',
    )
    # 调用
    response = _client.chat.completions.create(
        model="sam860/lucy:1.7b",
        messages=[{"role": "user", "content": "hello"}]
    )
    # 处理结果
    choice = response.choices[0]
    content = choice.message.content or ""
    print(f"LLM 回复：  {content}")


def test_1_invoke():
    """测试 1: 同步非流式调用"""
    banner("测试 1: invoke() - 同步非流式")
    t0 = time.time()
    llm = HelloAgentsLLM()
    print(f"[配置] model={llm.model}, adapter={type(llm._adapter).__name__}")

    response = llm.invoke(SYSTEM_PROMPT, temperature=0.3, max_tokens=MAX_TOKENS)
    elapsed = time.time() - t0

    print(f"[耗时]     {elapsed:.2f}s")
    print(f"[content]  {response.content!r}")
    print(f"[model]    {response.model}")
    print(f"[usage]    {response.usage}")
    print(f"[latency]  {response.latency_ms} ms")
    if response.reasoning_content:
        print(f"[reasoning] ({len(response.reasoning_content)} chars) "
              f"{response.reasoning_content[:100]}...")
    print(f"[repr]     {response!r}")

    assert response.content, "content 应非空"
    assert response.latency_ms > 0
    print(f"\n[PASS] test_1_invoke ({elapsed:.1f}s)")


def test_2_stream_invoke():
    """测试 2: 同步流式调用"""
    banner("测试 2: stream_invoke() - 同步流式")
    t0 = time.time()
    llm = HelloAgentsLLM()

    print("[流式输出] ", end="", flush=True)
    chunks = []
    for chunk in llm.stream_invoke(SIMPLE_PROMPT, temperature=0.3, max_tokens=MAX_TOKENS):
        chunks.append(chunk)
        # 实时打印（不阻塞）
        sys.stdout.write(chunk)
        sys.stdout.flush()

    full_text = "".join(chunks)
    elapsed = time.time() - t0

    print(f"\n[chunk 数]      {len(chunks)}")
    print(f"[完整文本]      {full_text!r}")
    if llm.last_call_stats:
        s = llm.last_call_stats
        print(f"[last_stats]    usage={s.usage}, latency={s.latency_ms}ms")
        if s.reasoning_content:
            print(f"[reasoning]     ({len(s.reasoning_content)} chars)")

    assert len(chunks) > 0, "应至少收到一个 chunk"
    assert llm.last_call_stats is not None, "应填充 last_call_stats"
    print(f"\n[PASS] test_2_stream_invoke ({elapsed:.1f}s)")


def test_3_invoke_with_tools():
    """测试 3: Function Calling"""
    banner("测试 3: invoke_with_tools() - Function Calling")
    t0 = time.time()
    llm = HelloAgentsLLM()

    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取城市天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }]

    response = llm.invoke_with_tools(
        TOOL_PROMPT, tools, tool_choice="auto",
        temperature=0.3, max_tokens=MAX_TOKENS
    )
    elapsed = time.time() - t0

    print(f"[耗时]     {elapsed:.2f}s")
    print(f"[content]  {response.content!r}")
    print(f"[tool_calls] {len(response.tool_calls)} 个")
    for i, tc in enumerate(response.tool_calls):
        print(f"  [{i}] name={tc.name}, args={tc.arguments}")
    print(f"[usage]    {response.usage}")
    print(f"[latency]  {response.latency_ms} ms")

    print(f"\n[PASS] test_3_invoke_with_tools ({elapsed:.1f}s)")


@pytest.mark.asyncio
async def test_4_ainvoke():
    """测试 4: 异步非流式"""
    banner("测试 4: ainvoke() - 异步非流式")
    t0 = time.time()
    llm = HelloAgentsLLM()

    response = await llm.ainvoke(ASYNC_PROMPT, temperature=0.3, max_tokens=MAX_TOKENS)
    elapsed = time.time() - t0

    print(f"[耗时]    {elapsed:.2f}s")
    print(f"[content] {response.content!r}")
    print(f"[usage]   {response.usage}")

    assert response.content
    print(f"\n[PASS] test_4_ainvoke ({elapsed:.1f}s)")


@pytest.mark.asyncio
async def test_5_astream_invoke():
    """测试 5: 异步流式"""
    banner("测试 5: astream_invoke() - 异步流式")
    t0 = time.time()
    llm = HelloAgentsLLM()

    print("[异步流式] ", end="", flush=True)
    chunks = []
    async for chunk in llm.astream_invoke(ASTREAM_PROMPT, temperature=0.3, max_tokens=MAX_TOKENS):
        chunks.append(chunk)
        sys.stdout.write(chunk)
        sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\n[chunk 数]   {len(chunks)}")
    print(f"[完整文本]   {''.join(chunks)!r}")
    if llm.last_call_stats:
        print(f"[last_stats] usage={llm.last_call_stats.usage}")

    assert len(chunks) > 0
    print(f"\n[PASS] test_5_astream_invoke ({elapsed:.1f}s)")


def main():
    print("=" * 60)
    print("  HelloAgentsLLM 基础测试")
    print(f"  模型: qwen3.5:latest (推理模型, max_tokens={MAX_TOKENS})")
    print("=" * 60)

    total_t0 = time.time()

    test_1_invoke()
    test_2_stream_invoke()
    test_3_invoke_with_tools()
    asyncio.run(test_4_ainvoke())
    asyncio.run(test_5_astream_invoke())

    total = time.time() - total_t0
    print("\n" + "=" * 60)
    print(f"  [ALL PASS] 5 个测试全部通过 (总耗时 {total:.1f}s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
