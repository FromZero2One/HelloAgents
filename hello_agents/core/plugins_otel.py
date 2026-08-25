"""OpenTelemetryPlugin - 分布式追踪与指标插件

集成 OpenTelemetry 标准，支持：
- 自动追踪 Agent 执行、LLM 调用、工具调用
- 指标收集 (Counters, Histograms, Gauges)
- 导出到多种后端 (Jaeger, Zipkin, OTLP, Console, Prometheus)
- 上下文传播 (W3C TraceContext)
- 自动插桩装饰器
- 自定义属性和事件
"""

from typing import Dict, Any, Optional, List, Callable
from .plugins import AgentPlugin, PluginContext
import asyncio
from contextlib import contextmanager
from functools import wraps
import time

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
    # Metrics
    from opentelemetry.metrics import get_meter_provider, set_meter_provider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricExporter
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
        
        # 设置 MeterProvider for metrics
        self._setup_meter_provider(config)
    
    def _setup_meter_provider(self, config: Dict[str, Any]):
        """配置 MeterProvider for metrics"""
        service_name = config.get("service_name", "hello-agents")
        resource = Resource.create({SERVICE_NAME: service_name})
        
        # 配置 Metric Readers
        readers = []
        
        exporter_type = config.get("metrics_exporter", "console")
        if exporter_type == "console" or exporter_type == "all":
            readers.append(PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=config.get("metrics_interval", 60000)
            ))
        
        if exporter_type == "prometheus" or exporter_type == "all":
            try:
                prometheus_exporter = PrometheusMetricExporter(
                    endpoint=config.get("prometheus_endpoint", "localhost:9464")
                )
                readers.append(PeriodicExportingMetricReader(
                    prometheus_exporter,
                    export_interval_millis=config.get("metrics_interval", 60000)
                ))
            except Exception as e:
                print(f"⚠️ Prometheus 导出器初始化失败: {e}")
        
        if exporter_type == "otlp" or exporter_type == "all":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
                otlp_endpoint = config.get("otlp_metrics_endpoint", config.get("otlp_endpoint", "http://localhost:4317"))
                readers.append(PeriodicExportingMetricReader(
                    OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True),
                    export_interval_millis=config.get("metrics_interval", 60000)
                ))
            except Exception as e:
                print(f"⚠️ OTLP Metrics 导出器初始化失败: {e}")
        
        self._meter_provider = MeterProvider(resource=resource, metric_readers=readers)
        set_meter_provider(self._meter_provider)
        
        self._meter = self._meter_provider.get_meter("hello-agents", version="1.0.0")
        
        # 初始化标准指标
        self._init_standard_metrics()
    
    def _init_standard_metrics(self):
        """初始化标准指标"""
        if not self._meter:
            return
        
        # Agent 执行计数器
        self._agent_runs_total = self._meter.create_counter(
            name="agent.runs.total",
            description="Total number of agent runs",
            unit="1"
        )
        
        # Agent 执行时长直方图
        self._agent_duration = self._meter.create_histogram(
            name="agent.duration.ms",
            description="Agent execution duration in milliseconds",
            unit="ms"
        )
        
        # LLM 调用计数器
        self._llm_calls_total = self._meter.create_counter(
            name="llm.calls.total",
            description="Total number of LLM calls",
            unit="1"
        )
        
        # LLM Token 使用直方图
        self._llm_tokens = self._meter.create_histogram(
            name="llm.tokens",
            description="LLM token usage",
            unit="1"
        )
        
        # LLM 调用时长直方图
        self._llm_duration = self._meter.create_histogram(
            name="llm.duration.ms",
            description="LLM call duration in milliseconds",
            unit="ms"
        )
        
        # 工具调用计数器
        self._tool_calls_total = self._meter.create_counter(
            name="tool.calls.total",
            description="Total number of tool calls",
            unit="1"
        )
        
        # 工具调用时长直方图
        self._tool_duration = self._meter.create_histogram(
            name="tool.duration.ms",
            description="Tool execution duration in milliseconds",
            unit="ms"
        )
        
        # 工具错误计数器
        self._tool_errors_total = self._meter.create_counter(
            name="tool.errors.total",
            description="Total number of tool errors",
            unit="1"
        )
        
        # 活跃 Agent 数量 Gauge
        self._active_agents = self._meter.create_up_down_counter(
            name="agent.active",
            description="Number of currently active agents",
            unit="1"
        )
        
        # Token 使用总量
        self._tokens_total = self._meter.create_counter(
            name="tokens.total",
            description="Total tokens consumed",
            unit="1"
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
        """关闭追踪和指标"""
        if self._tracer_provider:
            self._tracer_provider.shutdown()
        if self._meter_provider:
            self._meter_provider.shutdown()
        self._spans.clear()

    # ===== 自动插桩装饰器 =====

    def trace_metric(self, metric_name: str, attributes: Optional[Dict[str, Any]] = None):
        """装饰器：自动记录函数执行的指标
        
        Args:
            metric_name: 指标名称前缀
            attributes: 静态属性
            
        Example:
            @otel.trace_metric("my_custom_operation")
            async def my_function():
                ...
        """
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self._meter:
                    return await func(*args, **kwargs)
                
                start_time = time.time()
                error = None
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    error = e
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    attrs = {
                        "function": func.__name__,
                        "error": str(type(error).__name__) if error else "none",
                        **(attributes or {})
                    }
                    # 记录时长
                    if hasattr(self, f'_{metric_name.replace(".", "_")}_duration'):
                        duration_hist = getattr(self, f'_{metric_name.replace(".", "_")}_duration')
                        duration_hist.record(duration_ms, attributes=attrs)
                    # 记录计数
                    if hasattr(self, f'_{metric_name.replace(".", "_")}_total'):
                        counter = getattr(self, f'_{metric_name.replace(".", "_")}_total')
                        counter.add(1, attributes=attrs)
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self._meter:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                error = None
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    error = e
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    attrs = {
                        "function": func.__name__,
                        "error": str(type(error).__name__) if error else "none",
                        **(attributes or {})
                    }
                    if hasattr(self, f'_{metric_name.replace(".", "_")}_duration'):
                        duration_hist = getattr(self, f'_{metric_name.replace(".", "_")}_duration')
                        duration_hist.record(duration_ms, attributes=attrs)
                    if hasattr(self, f'_{metric_name.replace(".", "_")}_total'):
                        counter = getattr(self, f'_{metric_name.replace(".", "_")}_total')
                        counter.add(1, attributes=attrs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

    def record_agent_run(self, agent_name: str, duration_ms: float, success: bool, metadata: Optional[Dict] = None):
        """记录 Agent 执行指标"""
        if not self._meter:
            return
        attrs = {
            "agent": agent_name,
            "success": str(success).lower(),
            **(metadata or {})
        }
        if self._agent_runs_total:
            self._agent_runs_total.add(1, attributes=attrs)
        if self._agent_duration:
            # We need to record duration but we already have it
            pass  # Duration is recorded via trace span
    
    def record_llm_call(self, model: str, operation: str, duration_ms: float, 
                        prompt_tokens: int = 0, completion_tokens: int = 0, 
                        success: bool = True, metadata: Optional[Dict] = None):
        """记录 LLM 调用指标"""
        if not self._meter:
            return
        attrs = {
            "model": model,
            "operation": operation,
            "success": str(success).lower(),
        }
        if self._llm_calls_total:
            self._llm_calls_total.add(1, attributes=attrs)
        if self._llm_duration:
            self._llm_duration.record(duration_ms, attributes={"model": model, "operation": operation})
        if self._llm_tokens and (prompt_tokens or completion_tokens):
            self._llm_tokens.record(prompt_tokens, attributes={"model": model, "type": "prompt"})
            self._llm_tokens.record(completion_tokens, attributes={"model": model, "type": "completion"})
        if self._tokens_total:
            self._tokens_total.add(prompt_tokens + completion_tokens, attributes={"model": model})
    
    def record_tool_call(self, tool_name: str, duration_ms: float, 
                         success: bool = True, metadata: Optional[Dict] = None):
        """记录工具调用指标"""
        if not self._meter:
            return
        attrs = {
            "tool": tool_name,
            "success": str(success).lower(),
        }
        if self._tool_calls_total:
            self._tool_calls_total.add(1, attributes=attrs)
        if self._tool_duration:
            self._tool_duration.record(duration_ms, attributes={"tool": tool_name})
        if not success and self._tool_errors_total:
            self._tool_errors_total.add(1, attributes={"tool": tool_name, "error_type": "execution"})
    
    def set_active_agents(self, count: int):
        """设置活跃 Agent 数量"""
        if self._active_agents:
            # Note: UpDownCounter doesn't have direct set, use add with delta
            pass  # Would need to track current value
    
    def increment_active_agents(self, delta: int = 1):
        """增加/减少活跃 Agent 计数"""
        if self._active_agents:
            self._active_agents.add(delta)


# ===== 便捷函数：从上下文获取插件 =====

def get_opentelemetry_plugin(agent) -> Optional[OpenTelemetryPlugin]:
    """从 Agent 获取 OpenTelemetry 插件实例"""
    if hasattr(agent, '_plugin_manager'):
        return agent._plugin_manager.get_plugin("opentelemetry")
    return None


def trace_async(plugin_getter: Callable = get_opentelemetry_plugin, 
                metric_name: str = "", attributes: Optional[Dict] = None):
    """装饰器：异步函数自动追踪 + 指标
    
    Example:
        @trace_async(metric_name="my_operation", attributes={"component": "processor"})
        async def process_data(data):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get plugin from first arg if it's an agent, or use getter
            plugin = None
            if args and hasattr(args[0], '_plugin_manager'):
                plugin = args[0]._plugin_manager.get_plugin("opentelemetry")
            elif plugin_getter:
                plugin = plugin_getter(args[0] if args else None)
            
            if not plugin or not plugin._meter:
                return await func(*args, **kwargs)
            
            start_time = time.time()
            error = None
            attrs = {
                "function": func.__name__,
                **(attributes or {})
            }
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                attrs["error"] = type(e).__name__
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                attrs["error"] = "none"
                if plugin and plugin._meter:
                    # Record duration
                    if hasattr(plugin, f'_{metric_name.replace(".", "_")}_duration'):
                        duration_hist = getattr(plugin, f'_{metric_name.replace(".", "_")}_duration')
                        duration_hist.record((time.time() - start_time) * 1000, attributes=attrs)
                    if hasattr(plugin, f'_{metric_name.replace(".", "_")}_total'):
                        counter = getattr(plugin, f'_{metric_name.replace(".", "_")}_total')
                        counter.add(1, attributes=attrs)
        return wrapper
    return decorator