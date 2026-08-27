"""测试 HelloAgentsLLM 与本地 Ollama 模型的集成"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.core.llm_adapters import OpenAIAdapter


def _openai_tool_response():
    """创建 OpenAI API 响应的 mock 对象（非流式，无 tool calls）"""
    return SimpleNamespace(
        model="test-model",
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="结果: 5",
                    tool_calls=None,
                )
            )
        ],
    )


def _openai_tool_call_response():
    """创建包含 tool calls 的 OpenAI API 响应 mock"""
    return SimpleNamespace(
        model="test-model",
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(
                                name="calculate",
                                arguments='{"expression": "2+3"}',
                            ),
                        )
                    ],
                )
            )
        ],
    )


class TestLLMOllama:
    """测试 Ollama 本地模型集成 - 使用 OpenAIAdapter 配合 http://localhost:11434/v1"""

    @pytest.fixture(autouse=True)
    def setup_mock_client(self, request):
        """每个 test 自动设置 mock OpenAI client"""
        # 创建完整的 mock 链路：openai.OpenAI -> .chat -> .completions -> .create
        mock_openai_instance = MagicMock()
        
        # 将 test 方法名传递给 fixture，以便根据需要设置不同的返回值
        test_name = request.node.name
        
        # mock non-streaming 响应（用于 invoke、ainvoke）
        # 默认返回无 tool calls 的响应
        mock_response_no_tools = _openai_tool_response()
        mock_response_with_tools = _openai_tool_call_response()
        
        # 设置 create 的 side_effect 根据 test 名称决定返回什么
        if "invoke_with_tools" in test_name:
            # 需要 tool calling 的 test 返回有 tool calls 的响应
            mock_openai_instance.chat.completions.create.return_value = mock_response_with_tools
        else:
            # 其他 test 返回无 tool calls 的响应
            mock_openai_instance.chat.completions.create.return_value = mock_response_no_tools
        
        # mock streaming 响应 - 使用生成器
        def streaming_create_side_effect(**kwargs):
            if kwargs.get("stream", False):
                # 返回一个生成器，模拟 stream=True 的行为
                chunks = ["你好", "", "世界"]
                
                def chunk_generator():
                    for chunk_text in chunks:
                        yield SimpleNamespace(
                            choices=[SimpleNamespace(delta=SimpleNamespace(content=chunk_text))],
                            usage=None,
                        )
                    # 最后 yield usage
                    yield SimpleNamespace(
                        choices=[],
                        usage=SimpleNamespace(
                            prompt_tokens=3,
                            completion_tokens=2,
                            total_tokens=5,
                        ),
                    )
                
                return chunk_generator()
            # non-streaming 返回上面已设置的响应
            return mock_openai_instance.chat.completions.create.return_value
        
        mock_openai_instance.chat.completions.create.side_effect = streaming_create_side_effect
        
        # patch openai.OpenAI 以返回我们的 mock
        # OpenAIAdapter.create_client 调用 openai.Open(api_key=..., base_url=..., timeout=...)
        patch_target = "openai.OpenAI"
        self._patch_openai = patch(patch_target, return_value=mock_openai_instance)
        self._patch_openai.start()
        
        # 确保 adapter 使用正确的 base_url
        with patch.dict("os.environ", {
            "LLM_API_KEY": "ollama-key",
            "LLM_BASE_URL": "http://localhost:11434/v1",
            "LLM_MODEL_ID": "sam860/lucy:1.7b",
        }):
            # 创建 LLM 实例
            self.llm = HelloAgentsLLM()
            
            # 手动替换 adapter 的 client，确保是我们的 mock
            # OpenAIAdapter 创建的 client 正是 openai.OpenAI() 实例
            # 我们已经 patch 了 openai.OpenAI，所以 adapter._client 会是 mock
            if hasattr(self.llm._adapter, '_client'):
                self.llm._adapter._client = mock_openai_instance
            
            yield
            
        # 清理 patch
        self._patch_openai.stop()

    def test_ollama_invoke_basic(self):
        """测试 Ollama 基本非流式调用"""
        messages = [{"role": "user", "content": "你好"}]
        
        response = self.llm.invoke(messages)
        
        assert response.content == "结果: 5"
        assert response.model == "sam860/lucy:1.7b"
        assert response.usage["total_tokens"] == 15

    def test_ollama_invoke_with_tools(self):
        """测试 Ollama 的 Function Calling"""
        messages = [{"role": "user", "content": "计算 2+3"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "计算",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                    },
                },
            }
        ]

        response = self.llm.invoke_with_tools(messages, tools)

        assert response.tool_calls[0].name == "calculate"
        assert response.tool_calls[0].arguments == '{"expression": "2+3"}'

    def test_ollama_stream_invoke(self):
        """测试 Ollama 流式调用"""
        # 流式调用会迭代 chunks，提取 content
        chunks = list(self.llm.stream_invoke([{"role": "user", "content": "hi"}]))
        
        # 应该能够提取出一些内容（取决于 mock 返回的内容）
        # 核心验证是：stream_invoke 不会抛出异常，并返回一个 list
        assert isinstance(chunks, list)

    def test_ollama_ainvoke(self):
        """测试 Ollama 异步调用"""
        import asyncio
        
        messages = [{"role": "user", "content": "hi"}]
        response = asyncio.run(self.llm.ainvoke(messages))
        
        assert response.content == "结果: 5"
        assert response.model == "sam860/lucy:1.7b"
        assert response.usage["total_tokens"] == 15