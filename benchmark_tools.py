"""Performance Benchmark for Concurrent Tool Execution

测试场景：
1. 串行 vs 并行工具执行对比
2. 不同并发数下的吞吐量
3. 工具执行延迟分布
4. 内存/CPU 使用情况
"""

import asyncio
import time
import statistics
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hello_agents.core.config import Config
from hello_agents.core.plugins_tool import ToolPlugin
from hello_agents.core.plugins import PluginContext
from hello_agents.tools.registry import ToolRegistry
from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.response import ToolResponse, ToolStatus
from hello_agents.core.agent import Agent


# ==================== 测试工具 ====================

class MockAsyncTool(Tool):
    """模拟异步工具 - 可配置延迟"""
    
    def __init__(self, name: str, delay_ms: int = 100, should_fail: bool = False):
        super().__init__(
            name=name,
            description=f"Mock tool with {delay_ms}ms delay"
        )
        self.base_delay = delay_ms / 1000.0  # 转为秒
        self.should_fail = should_fail
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="input", type="string", required=True, description="输入"),
            ToolParameter(name="delay_override", type="integer", required=False, description="覆盖延迟(ms)"),
        ]
    
    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        delay = parameters.get("delay_override", self.base_delay * 1000) / 1000.0
        time.sleep(delay)
        
        if self.should_fail:
            return ToolResponse.error("MOCK_ERROR", "Mock failure")
        
        return ToolResponse.success(f"Result from {self.name} after {delay*1000:.0f}ms")
    
    async def arun(self, parameters: Dict[str, Any]) -> ToolResponse:
        delay = parameters.get("delay_override", self.base_delay * 1000) / 1000.0
        await asyncio.sleep(delay)
        
        if self.should_fail:
            return ToolResponse.error("MOCK_ERROR", "Mock failure")
        
        return ToolResponse.success(f"Result from {self.name} after {delay*1000:.0f}ms")


class CPUBoundTool(Tool):
    """CPU 密集型工具 - 模拟计算任务"""
    
    def __init__(self, name: str, iterations: int = 100000):
        super().__init__(
            name=name,
            description=f"CPU bound tool with {iterations} iterations"
        )
        self.iterations = iterations
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="input", type="string", required=True),
        ]
    
    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        # CPU 密集计算
        result = 0
        for i in range(self.iterations):
            result += i * i
        return ToolResponse.success(f"Computed sum of squares: {result}")
    
    async def arun(self, parameters: Dict[str, Any]) -> ToolResponse:
        # 在线程池中运行 CPU 密集任务
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.run(parameters))


# ==================== 基准测试框架 ====================

@dataclass
class BenchmarkResult:
    """单次基准测试结果"""
    name: str
    mode: str  # "serial" | "parallel"
    concurrency: int
    tool_count: int
    total_time_ms: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_per_sec: float
    success_count: int
    error_count: int
    latencies_ms: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.results: List[BenchmarkResult] = []
    
    def _create_tool_plugin(self, concurrent: int, tools: List[Tool] = None) -> ToolPlugin:
        """创建配置了指定并发数的 ToolPlugin，并注册工具"""
        config = Config(max_concurrent_tools=concurrent)
        plugin = ToolPlugin(config)
        # Initialize the plugin to create tool_registry
        from hello_agents.core.plugins import PluginContext
        
        # Create a minimal context for initialization
        class MockAgent:
            name = "benchmark"
        
        mock_agent = MockAgent()
        context = PluginContext(
            agent=mock_agent,
            config=config,
            llm=None,
            tool_registry=None
        )
        plugin.setup(context)
        
        if tools:
            for tool in tools:
                plugin.tool_registry.register_tool(tool)
        return plugin
    
    def _create_mock_tools(self, count: int, delay_ms: int = 50) -> List[MockAsyncTool]:
        """创建模拟工具"""
        return [MockAsyncTool(f"tool_{i}", delay_ms=delay_ms) for i in range(count)]
    
    async def run_serial(self, tool_plugin: ToolPlugin, tools: List[Tool], 
                         tool_args: List[Dict]) -> BenchmarkResult:
        """串行执行基准测试"""
        latencies = []
        success = 0
        errors = 0
        
        start = time.perf_counter()
        
        for tool, args in zip(tools, tool_args):
            tool_start = time.perf_counter()
            try:
                response = await tool_plugin.aexecute_tool_call(tool.name, args)
                if response.status == ToolStatus.SUCCESS:
                    success += 1
                else:
                    errors += 1
            except Exception as e:
                errors += 1
            finally:
                latencies.append((time.perf_counter() - tool_start) * 1000)
        
        total_time = (time.perf_counter() - start) * 1000
        
        return BenchmarkResult(
            name=f"serial_{len(tools)}_tools",
            mode="serial",
            concurrency=1,
            tool_count=len(tools),
            total_time_ms=total_time,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p50_latency_ms=statistics.median(latencies) if latencies else 0,
            p95_latency_ms=self._percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=self._percentile(latencies, 99) if latencies else 0,
            throughput_per_sec=len(tools) / (total_time / 1000) if total_time > 0 else 0,
            success_count=success,
            error_count=errors,
            latencies_ms=latencies
        )
    
    async def run_parallel(self, tool_plugin: ToolPlugin, tools: List[Tool], 
                          tool_args: List[Dict]) -> BenchmarkResult:
        """并行执行基准测试"""
        latencies = []
        success = 0
        errors = 0
        
        # 准备工具调用列表
        tool_calls = [
            {"name": tool.name, "arguments": args, "id": f"call_{i}"}
            for i, (tool, args) in enumerate(zip(tools, tool_args))
        ]
        
        start = time.perf_counter()
        
        try:
            results = await tool_plugin.aexecute_tool_calls_parallel(tool_calls)
            
            for i, (tool, args, result) in enumerate(zip(tools, tool_args, results)):
                # 计算延迟（这里无法精确测量单个工具延迟，使用总时间均分近似）
                latencies.append((time.perf_counter() - start) * 1000 / len(tools))
                
                if result.status == ToolStatus.SUCCESS:
                    success += 1
                else:
                    errors += 1
                    
        except Exception as e:
            print(f"Parallel execution error: {e}")
            errors = len(tools)
        
        total_time = (time.perf_counter() - start) * 1000
        
        return BenchmarkResult(
            name=f"parallel_{len(tools)}_tools_c{tool_plugin._get_semaphore()._value if hasattr(tool_plugin, '_tool_semaphore') else 'unknown'}",
            mode="parallel",
            concurrency=tool_plugin._get_semaphore()._value if hasattr(tool_plugin, '_tool_semaphore') and tool_plugin._tool_semaphore else 0,
            tool_count=len(tools),
            total_time_ms=total_time,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p50_latency_ms=statistics.median(latencies) if latencies else 0,
            p95_latency_ms=self._percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=self._percentile(latencies, 99) if latencies else 0,
            throughput_per_sec=len(tools) / (total_time / 1000) if total_time > 0 else 0,
            success_count=success,
            error_count=errors,
            latencies_ms=latencies
        )
    
    def _percentile(self, data: List[float], p: int) -> float:
        if not data:
            return 0
        sorted_data = sorted(data)
        k = int(len(sorted_data) * p / 100)
        return sorted_data[min(k, len(sorted_data) - 1)]
    
    async def run_benchmark_suite(self) -> List[BenchmarkResult]:
        """运行完整基准测试套件"""
        print("Starting Performance Benchmark Suite")
        print("=" * 60)
        
        test_configs = [
            # (tool_count, delay_ms, concurrency_levels)
            (5, 100, [1, 2, 3, 5]),
            (10, 50, [1, 2, 3, 5, 10]),
            (20, 20, [1, 2, 3, 5, 10]),
        ]
        
        for tool_count, delay_ms, concurrency_levels in test_configs:
            print(f"\nTesting {tool_count} tools with {delay_ms}ms delay")
            
            tools = self._create_mock_tools(tool_count, delay_ms)
            tool_args = [{"input": f"test_input_{i}"} for i in range(tool_count)]
            
            # 串行基准（并发=1）
            serial_plugin = self._create_tool_plugin(1, tools)
            serial_result = await self.run_serial(serial_plugin, tools, tool_args)
            self.results.append(serial_result)
            print(f"  Serial: {serial_result.total_time_ms:.1f}ms, "
                  f"throughput: {serial_result.throughput_per_sec:.1f} ops/s")
            
            # 并行基准（不同并发级别）
            for concurrency in concurrency_levels:
                if concurrency == 1:
                    continue
                
                parallel_plugin = self._create_tool_plugin(concurrency, tools)
                parallel_result = await self.run_parallel(parallel_plugin, tools, tool_args)
                self.results.append(parallel_result)
                
                speedup = serial_result.total_time_ms / parallel_result.total_time_ms if parallel_result.total_time_ms > 0 else 0
                print(f"  Parallel(c={concurrency}): {parallel_result.total_time_ms:.1f}ms, "
                      f"throughput: {parallel_result.throughput_per_sec:.1f} ops/s, "
                      f"speedup: {speedup:.2f}x")
        
        return self.results
    
    def print_summary(self):
        """打印汇总报告"""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        # 按工具数量分组
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in self.results:
            grouped[r.tool_count].append(r)
        
        for tool_count, results in sorted(grouped.items()):
            print(f"\n{tool_count} Tools:")
            print(f"  {'Mode':<12} {'Concurrency':<12} {'Time(ms)':<10} {'Throughput':<12} {'Speedup':<8} {'P50':<8} {'P95':<8}")
            print(f"  {'-'*70}")
            
            serial_result = next((r for r in results if r.mode == "serial"), None)
            
            for r in results:
                speedup = ""
                if serial_result and r.mode == "parallel":
                    speedup = f"{serial_result.total_time_ms / r.total_time_ms:.2f}x" if r.total_time_ms > 0 else "N/A"
                
                print(f"  {r.mode:<12} {r.concurrency:<12} {r.total_time_ms:<10.1f} "
                      f"{r.throughput_per_sec:<12.1f} {speedup:<8} "
                      f"{r.p50_latency_ms:<8.1f} {r.p95_latency_ms:<8.1f}")
    
    def save_results(self, filepath: str):
        """保存结果到 JSON"""
        data = {
            "timestamp": time.time(),
            "results": [
                {
                    "name": r.name,
                    "mode": r.mode,
                    "concurrency": r.concurrency,
                    "tool_count": r.tool_count,
                    "total_time_ms": r.total_time_ms,
                    "avg_latency_ms": r.avg_latency_ms,
                    "p50_latency_ms": r.p50_latency_ms,
                    "p95_latency_ms": r.p95_latency_ms,
                    "p99_latency_ms": r.p99_latency_ms,
                    "throughput_per_sec": r.throughput_per_sec,
                    "success_count": r.success_count,
                    "error_count": r.error_count,
                }
                for r in self.results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {filepath}")


# ==================== CPU 密集型工具测试 ====================

async def benchmark_cpu_tools():
    """测试 CPU 密集型工具的并行执行（使用线程池）"""
    print("\nCPU-Bound Tool Benchmark (ThreadPoolExecutor)")
    print("-" * 50)
    
    config = Config(max_concurrent_tools=4)
    tool_plugin = ToolPlugin(config)
    
    # Initialize the plugin
    from hello_agents.core.plugins import PluginContext
    
    class MockAgent:
        name = "benchmark"
    
    mock_agent = MockAgent()
    context = PluginContext(
        agent=mock_agent,
        config=config,
        llm=None,
        tool_registry=None
    )
    tool_plugin.setup(context)
    
    cpu_tools = [CPUBoundTool(f"cpu_{i}", iterations=500000) for i in range(4)]
    # Register tools
    for tool in cpu_tools:
        tool_plugin.tool_registry.register_tool(tool)
    
    tool_args = [{"input": f"data_{i}"} for i in range(4)]
    
    # 串行
    start = time.perf_counter()
    for tool, args in zip(cpu_tools, tool_args):
        await tool_plugin.aexecute_tool_call(tool.name, args)
    serial_time = (time.perf_counter() - start) * 1000
    
    # 并行
    tool_calls = [
        {"name": t.name, "arguments": a, "id": f"call_{i}"}
        for i, (t, a) in enumerate(zip(cpu_tools, tool_args))
    ]
    
    start = time.perf_counter()
    results = await tool_plugin.aexecute_tool_calls_parallel(tool_calls)
    parallel_time = (time.perf_counter() - start) * 1000
    
    speedup = serial_time / parallel_time if parallel_time > 0 else 0
    
    print(f"  Serial:   {serial_time:.1f}ms")
    print(f"  Parallel: {parallel_time:.1f}ms (c=4)")
    print(f"  Speedup:  {speedup:.2f}x")
    
    return serial_time, parallel_time, speedup


# ==================== 主函数 ====================

async def main():
    """主函数"""
    print("HelloAgents Concurrent Tool Execution Benchmark")
    print("=" * 60)
    
    runner = BenchmarkRunner()
    
    # 运行 I/O 密集型工具基准
    await runner.run_benchmark_suite()
    runner.print_summary()
    
    # 运行 CPU 密集型工具基准
    await benchmark_cpu_tools()
    
    # 保存结果
    runner.save_results("benchmark_results.json")
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    asyncio.run(main())