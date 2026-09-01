"""测试 HelloAgentsLLM 与本地 Ollama 模型的真实集成（非 mock）

环境要求：
- 本地 Ollama 服务运行在 http://localhost:11434
- 已拉取模型（默认 qwen3.5:latest，从 .env 的 LLM_MODEL_ID 读取）

覆盖范围：
  1. test_ollama_invoke_basic      - 同步非流式
  2. test_ollama_invoke_with_tools - Function Calling
  3. test_ollama_stream_invoke     - 同步流式
  4. test_ollama_ainvoke           - 异步非流式

说明: qwen3.5 是推理模型，会先输出 thinking 再给答案，
      所有测试统一 max_tokens 预算 + 极简 prompt，控制单次 5-15s。
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

# 手动加载 .env（LLM_MODEL_ID 等配置）
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from hello_agents.core.llm import HelloAgentsLLM

# 本地 Ollama OpenAI 兼容端点
OLLAMA_BASE_URL = "http://localhost:11434/v1"
# 推理模型 thinking 占大头，需要给足 token 预算
MAX_TOKENS = 32 * 1024


@pytest.fixture()
def llm():
    """创建指向本地 Ollama 的 LLM 实例（模型从 .env 的 LLM_MODEL_ID 读取）"""
    return HelloAgentsLLM(base_url=OLLAMA_BASE_URL, timeout=120)


class TestLLMOllama:
    """测试本地 Ollama 真实集成（localhost:11434）"""

    def test_ollama_invoke_basic(self, llm):
        """Ollama 基本非流式调用"""
        t0 = time.time()
        response = llm.invoke(
            [{"role": "user", "content": "1+1=?"}],
            temperature=0.3, max_tokens=MAX_TOKENS
        )
        elapsed = time.time() - t0

        print(f"[耗时]     {elapsed:.2f}s")
        print(f"[content]  {response.content!r}")
        print(f"[model]    {response.model}")
        print(f"[usage]    {response.usage}")
        print(f"[latency]  {response.latency_ms} ms")
        if response.reasoning_content:
            print(f"[reasoning] ({len(response.reasoning_content)} chars)")

        assert response.content, "content 应非空"
        assert response.model == llm.model, "应返回配置的模型名"
        assert response.usage["total_tokens"] > 0, "应返回 token 用量"
        print(f"\n[PASS] test_ollama_invoke_basic ({elapsed:.1f}s)")

    def test_ollama_invoke_with_tools(self, llm):
        """Ollama 的 Function Calling"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定城市的天气",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string", "description": "城市名"}},
                        "required": ["city"],
                    },
                },
            }
        ]

        t0 = time.time()
        response = llm.invoke_with_tools(
            [{"role": "user", "content": "查询北京的天气"}],
            tools,
            tool_choice="auto",
            temperature=0.3, max_tokens=MAX_TOKENS
        )
        elapsed = time.time() - t0

        print(f"[耗时]       {elapsed:.2f}s")
        print(f"[content]    {response.content!r}")
        print(f"[tool_calls] {len(response.tool_calls)} 个")
        for i, tc in enumerate(response.tool_calls):
            print(f"  [{i}] name={tc.name}, args={tc.arguments}")
        print(f"[usage]      {response.usage}")

        # 真实模型行为不确定：可能调用工具，也可能直接回答；
        # 但至少应返回有效响应（Function Calling 链路无异常）
        assert response.content or response.tool_calls, "应返回文本或工具调用"
        for tc in response.tool_calls:
            assert tc.name == "get_weather", "工具名应匹配"
            assert '"city"' in tc.arguments, "参数应包含 city"
        print(f"\n[PASS] test_ollama_invoke_with_tools ({elapsed:.1f}s)")

    def test_ollama_stream_invoke(self, llm):
        """Ollama 流式调用"""
        t0 = time.time()
        chunks = list(llm.stream_invoke(
            [{"role": "user", "content": "说hi"}],
            temperature=0.3, max_tokens=MAX_TOKENS
        ))
        elapsed = time.time() - t0

        full_text = "".join(chunks)
        print(f"[耗时]     {elapsed:.2f}s")
        print(f"[chunk 数] {len(chunks)}")
        print(f"[完整文本] {full_text!r}")

        assert len(chunks) > 0, "应至少收到一个 chunk"
        assert full_text.strip(), "流式文本应非空"
        print(f"\n[PASS] test_ollama_stream_invoke ({elapsed:.1f}s)")

    @pytest.mark.asyncio
    async def test_ollama_ainvoke(self, llm):
        """Ollama 异步调用"""
        t0 = time.time()
        response = await llm.ainvoke(
            [{"role": "user", "content": "1+1=?"}],
            temperature=0.3, max_tokens=MAX_TOKENS
        )
        elapsed = time.time() - t0

        print(f"[耗时]    {elapsed:.2f}s")
        print(f"[content] {response.content!r}")
        print(f"[model]   {response.model}")
        print(f"[usage]   {response.usage}")

        assert response.content, "content 应非空"
        assert response.model == llm.model
        print(f"\n[PASS] test_ollama_ainvoke ({elapsed:.1f}s)")


def main():
    """直接运行入口（无需 pytest）"""
    print("=" * 60)
    print("  HelloAgentsLLM x Ollama 集成测试（真实调用）")
    print(f"  base_url: {OLLAMA_BASE_URL}")
    print("=" * 60)

    llm = HelloAgentsLLM(base_url=OLLAMA_BASE_URL, timeout=120)
    test = TestLLMOllama()

    test.test_ollama_invoke_basic(llm)
    test.test_ollama_invoke_with_tools(llm)
    test.test_ollama_stream_invoke(llm)
    asyncio.run(test.test_ollama_ainvoke(llm))


if __name__ == "__main__":
    main()
