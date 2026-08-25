"""OpenTelemetryPlugin - 分布式追踪插件

集成 OpenTelemetry 标准，支持：
- 自动追踪 Agent 执行、LLM 调用、工具调用
- 导出到多种后端 (Jaeger, Zipkin, OTLP, Console)
- 上下文传播 (W3C TraceContext)
- 自定义属性和事件
"""

from typing import Dict, Any, Optional, List
from .plugins import AgentPlugin, PluginContext
import asyncio
from contextlib import contextmanager

# Optional imports - graceful degradation if not installed
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.jaeger.proto.grpc import JaegerExporter
    from opentelemetry.exporter.zipkin.proto.http import ZipkinExporter
    from opentelemetry.propagate import inject, extract
    from opentelemetry.trace import SpanKind, Status, StatusCode
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None


class TelemetryExporterType(str):
    """导出器类型"""
    CONSOLE = "console"
    OTLP = "otlp"
    JAEGER = "jaeger"
    ZIPKIN = "zipkin"


class OpenTelemetryPlugin(AgentPlugin):
    """OpenTelemetry 分布式追踪插件"""
    
    name = "opentelemetry"
    priority = 15  # 早期初始化，供其他插件使用
    
    def __init__(self, config=None):
        super().__init__(config)
        self._tracer_provider: Optional[Any] = None
        self._tracer: Optional[Any] = None
        self._spans: Dict[str, Any] = {}  # 活跃 span 追踪
        self._enabled = False
    
    def _initialize(self) -> None:
        if not OTEL_AVAILABLE:
            if self.config.debug:
                print("⚠️ OpenTelemetry 未安装，跳过初始化 (pip install opentelemetry-sdk opentelemetry-exporter-otlp)")
            return
        
        # 检查配置是否启用
        otel_config = getattr(self.config, 'opentelemetry', None)
        if not otel_config:
            return
        
        self._enabled = otel_config.get("enabled", True)
        if not self._enabled:
            return
        
        self._setup_tracer_provider(otel_config)
        self._setup_exporters(otel_config)
        
        # 自动插桩 HTTP 请求
        if otel_config.get("instrument_requests", True):
            RequestsInstrumentor().instrument()
        
        # 注入到上下文供其他组件使用
        self.context.opentelemetry_tracer = self._tracer
        self.context.opentelemetry_propagate = self._propagate_context
        
        if self.config.debug:
            print(f"✅ OpenTelemetry 初始化完成: {otel_config.get('exporter', 'console')}")
    
    def _setup_tracer_provider(self, config: Dict[str, Any]):
        """配置 TracerProvider"""
        service_name = config.get("service_name", "hello-agents")
        resource = Resource.create({SERVICE_NAME: service_name})
        
        self._tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(self._tracer_provider)
        
        self._tracer = trace.get_tracer(
            "hello-agents",
            version="1.0.0"
        )
    
    def _setup_exporters(self, config: Dict[str, Any]):
        """配置导出器"""
        exporter_type = config.get("exporter", "console")
        exporters = []
        
        if exporter_type == "console" or exporter_type == "all":
            exporters.append(ConsoleSpanExporter())
        
        if exporter_type == "otlp" or exporter_type == "all":
            endpoint = config.get("otlp_endpoint", "http://localhost:4317")
            try:
                exporters.append(OTLPSpanExporter(endpoint=endpoint, insecure=True))
            except Exception as e:
                print(f"⚠️ OTLP 导出器初始化失败: {e}")
        
        if exporter_type == "jaeger" or exporter_type == "all":
            jaeger_endpoint = config.get("jaeger_endpoint", "http://localhost:14250")
            try:
                exporters.append(JaegerExporter(
                    agent_host_name=jaeger_endpoint.replace("http://", "").split(":")[0],
                    agent_port=int(jaeger_endpoint.split(":")[-1]) if ":" in jaeger_endpoint else 14250,
                ))
            except Exception as e:
                print(f"⚠️ Jaeger 导出器初始化失败: {e}")
        
        if exporter_type == "zipkin" or exporter_type == "all":
            zipkin_endpoint = config.get("zipkin_endpoint", "http://localhost:9411/api/v2/spans")
            try:
                exporters.append(ZipkinExporter(endpoint=zipkin_endpoint))
            except Exception as e:
                print(f"⚠️ Zipkin 导出器初始化失败: {e}")
        
        # 添加批处理器
        for exporter in exporters:
            self._tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    
    def _propagate_context(self, carrier: Dict[str, str]) -> Dict[str, str]:
        """注入追踪上下文到 carrier (用于 HTTP 头传播)"""
        if not OTEL_AVAILABLE or not self._enabled:
            return carrier
        inject(carrier)
        return carrier
    
    def extract_context(self, carrier: Dict[str, str]):
        """从 carrier 提取追踪上下文"""
        if not OTEL_AVAILABLE or not self._enabled:
            return None
        return extract(carrier)
    
    # ===== 公共 API =====
    
    @contextmanager
    def start_span(
        self,
        name: str,
        kind: str = "internal",
        attributes: Optional[Dict[str, Any]] = None,
        parent_context: Optional[Any] = None
    ):
        """启动一个新的 span (上下文管理器)
        
        Args:
            name: Span 名称
            kind: SpanKind (internal, server, client, producer, consumer)
            attributes: 初始属性
            parent_context: 父上下文
            
        Example:
            with otel.start_span("llm.call") as span:
                span.set_attribute("model", "gpt-4")
                result = await llm.ainvoke(messages)
        """
        if not OTEL_AVAILABLE or not self._enabled or not self._tracer:
            yield None
            return
        
        span_kind = getattr(SpanKind, kind.upper(), SpanKind.INTERNAL)
        
        with self._tracer.start_as_current_span(
            name,
            kind=span_kind,
            context=parent_context,
            attributes=attributes or {}
        ) as span:
            # 记录 span 以便后续管理
            span_id = span.get_span_context().span_id
            self._spans[str(span_id)] = span
            try:
                yield span
            finally:
                self._spans.pop(str(span_id), None)
    
    def start_span_manual(
        self,
        name: str,
        kind: str = "internal",
        attributes: Optional[Dict[str, Any]] = None,
        parent_context: Optional[Any] = None
    ) -> Optional[Any]:
        """手动启动 span (需手动 end)
        
        Returns:
            Span 对象，需调用 span.end()
        """
        if not OTEL_AVAILABLE or not self._enabled or not self._tracer:
            return None
        
        span_kind = getattr(SpanKind, kind.upper(), SpanKind.INTERNAL)
        
        span = self._tracer.start_span(
            name,
            kind=span_kind,
            context=parent_context,
            attributes=attributes or {}
        )
        
        span_id = span.get_span_context().span_id
        self._spans[str(span_id)] = span
        return span
    
    def end_span(self, span: Any, status: str = "ok", error: Optional[Exception] = None):
        """结束 span"""
        if not span:
            return
        
        if error:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
        else:
            span.set_status(Status(StatusCode.OK))
        
        span.end()
        span_id = span.get_span_context().span_id
        self._spans.pop(str(span_id), None)
    
    def add_event(self, span: Any, name: str, attributes: Optional[Dict[str, Any]] = None):
        """添加事件到 span"""
        if span and OTEL_AVAILABLE:
            span.add_event(name, attributes or {})
    
    def set_attribute(self, span: Any, key: str, value: Any):
        """设置 span 属性"""
        if span and OTEL_AVAILABLE:
            span.set_attribute(key, value)
    
    # ===== 便捷方法：Agent/工具/LLM 专用 =====
    
    @contextmanager
    def trace_agent_run(self, agent_name: str, input_text: str, metadata: Optional[Dict] = None):
        """追踪 Agent 完整执行"""
        attrs = {
            "agent.name": agent_name,
            "agent.input_length": len(input_text),
            **(metadata or {})
        }
        with self.start_span(f"agent.{agent_name}.run", kind="server", attributes=attrs) as span:
            yield span
    
    @contextmanager
    def trace_llm_call(self, model: str, operation: str = "invoke", metadata: Optional[Dict] = None):
        """追踪 LLM 调用"""
        attrs = {
            "llm.model": model,
            "llm.operation": operation,
            **(metadata or {})
        }
        with self.start_span(f"llm.{operation}", kind="client", attributes=attrs) as span:
            yield span
    
    @contextmanager
    def trace_tool_call(self, tool_name: str, metadata: Optional[Dict] = None):
        """追踪工具调用"""
        attrs = {
            "tool.name": tool_name,
            **(metadata or {})
        }
        with self.start_span(f"tool.{tool_name}", kind="client", attributes=attrs) as span:
            yield span
    
    def get_active_spans_count(self) -> int:
        return len(self._spans)
    
    async def teardown(self):
        """关闭追踪"""
        if self._tracer_provider:
            self._tracer_provider.shutdown()
        self._spans.clear()


# 如果 OpenTelemetry 不可用，提供空实现
if not OTEL_AVAILABLE:
    class OpenTelemetryPlugin(AgentPlugin):
        """空实现 - OpenTelemetry 未安装时使用"""
        
        name = "opentelemetry"
        priority = 15
        
        def _initialize(self):
            if self.config.debug:
                print("⚠️ OpenTelemetry 未安装，追踪功能不可用")
        
        @contextmanager
        def start_span(self, *args, **kwargs):
            yield None
        
        def start_span_manual(self, *args, **kwargs):
            return None
        
        def end_span(self, *args, **kwargs):
            pass
        
        def add_event(self, *args, **kwargs):
            pass
        
        def set_attribute(self, *args, **kwargs):
            pass
        
        @contextmanager
        def trace_agent_run(self, *args, **kwargs):
            yield None
        
        @contextmanager
        def trace_llm_call(self, *args, **kwargs):
            yield None
        
        @contextmanager
        def trace_tool_call(self, *args, **kwargs):
            yield None
        
        def get_active_spans_count(self):
            return 0
        
        async def teardown(self):
            pass