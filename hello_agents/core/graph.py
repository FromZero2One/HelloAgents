"""AgentGraph - 多 Agent 编排引擎

设计目标：
- 支持有向无环图（DAG）的 Agent 编排
- 类 LangGraph API：节点、边、条件分支、状态管理
- 内置并行执行、错误重试、检查点
- 与现有 Agent/Plugin 系统无缝集成
"""

from typing import Dict, List, Any, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import uuid
from datetime import datetime
import json
from collections import deque


class NodeStatus(Enum):
    """节点执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class GraphState:
    """图执行状态 - 在节点间传递的共享状态"""
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
    
    def update(self, other: Dict[str, Any]) -> None:
        self.data.update(other)
    
    def copy(self) -> 'GraphState':
        return GraphState(
            data=self.data.copy(),
            metadata=self.metadata.copy()
        )


class GraphNode(ABC):
    """图节点基类"""
    
    def __init__(
        self,
        name: str,
        agent: Optional['Agent'] = None,
        func: Optional[Callable] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.agent = agent
        self.func = func
        self.config = config or {}
        self.status = NodeStatus.PENDING
        self.result: Any = None
        self.error: Optional[Exception] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    @abstractmethod
    async def execute(self, state: GraphState) -> GraphState:
        """执行节点逻辑
        
        Args:
            state: 输入状态
            
        Returns:
            更新后的状态
        """
        pass
    
    def _mark_started(self):
        self.status = NodeStatus.RUNNING
        self.start_time = datetime.now()
    
    def _mark_completed(self, result: Any):
        self.status = NodeStatus.COMPLETED
        self.result = result
        self.end_time = datetime.now()
    
    def _mark_failed(self, error: Exception):
        self.status = NodeStatus.FAILED
        self.error = error
        self.end_time = datetime.now()
    
    def _mark_skipped(self):
        self.status = NodeStatus.SKIPPED
        self.end_time = datetime.now()
    
    @property
    def duration_ms(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None


class AgentNode(GraphNode):
    """Agent 节点 - 使用现有 Agent 执行"""
    
    def __init__(
        self,
        name: str,
        agent: 'Agent',
        input_key: str = "input",
        output_key: str = "output",
        **kwargs
    ):
        super().__init__(name, agent=agent, **kwargs)
        self.input_key = input_key
        self.output_key = output_key
    
    async def execute(self, state: GraphState) -> GraphState:
        self._mark_started()
        
        try:
            input_text = state.get(self.input_key, "")
            
            # 使用 Agent 的异步执行
            if hasattr(self.agent, '_arun_impl'):
                result = await self.agent._arun_impl(input_text)
            elif hasattr(self.agent, 'arun'):
                result = await self.agent.arun(input_text)
            else:
                # 回退到同步执行
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: self.agent.run(input_text)
                )
            
            # 更新状态
            new_state = state.copy()
            new_state.set(self.output_key, result)
            
            self._mark_completed(result)
            return new_state
            
        except Exception as e:
            self._mark_failed(e)
            raise


class FunctionNode(GraphNode):
    """函数节点 - 执行自定义函数"""
    
    def __init__(
        self,
        name: str,
        func: Callable[[GraphState], Union[GraphState, Any]],
        **kwargs
    ):
        super().__init__(name, func=func, **kwargs)
    
    async def execute(self, state: GraphState) -> GraphState:
        self._mark_started()
        
        try:
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(state)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: self.func(state))
            
            # 支持返回新状态或直接返回值
            if isinstance(result, GraphState):
                new_state = result
            else:
                new_state = state.copy()
                new_state.set(self.name, result)
            
            self._mark_completed(result)
            return new_state
            
        except Exception as e:
            self._mark_failed(e)
            raise


class ConditionNode(GraphNode):
    """条件节点 - 基于状态决定下一步"""
    
    def __init__(
        self,
        name: str,
        condition: Callable[[GraphState], bool],
        true_branch: str,
        false_branch: str,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch
    
    async def execute(self, state: GraphState) -> GraphState:
        self._mark_started()
        
        try:
            result = self.condition(state)
            branch = self.true_branch if result else self.false_branch
            
            new_state = state.copy()
            new_state.set("_next_node", branch)
            new_state.set(f"{self.name}_condition_result", result)
            
            self._mark_completed(branch)
            return new_state
            
        except Exception as e:
            self._mark_failed(e)
            raise


class ParallelNode(GraphNode):
    """并行节点 - 同时执行多个子节点"""
    
    def __init__(
        self,
        name: str,
        nodes: List[GraphNode],
        merge_strategy: str = "all",  # all, any, first
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.nodes = nodes
        self.merge_strategy = merge_strategy
    
    async def execute(self, state: GraphState) -> GraphState:
        self._mark_started()
        
        try:
            # 并行执行所有子节点
            tasks = [node.execute(state.copy()) for node in self.nodes]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            new_state = state.copy()
            successful_results = []
            errors = []
            
            for i, (node, result) in enumerate(zip(self.nodes, results)):
                if isinstance(result, Exception):
                    errors.append({"node": node.name, "error": str(result)})
                    node._mark_failed(result)
                elif isinstance(result, GraphState):
                    successful_results.append(result)
                    node._mark_completed(result)
                    # 合并状态
                    new_state.data.update(result.data)
                else:
                    successful_results.append(result)
                    node._mark_completed(result)
            
            # 根据合并策略决定结果
            if self.merge_strategy == "all" and errors:
                raise Exception(f"Parallel nodes failed: {errors}")
            elif self.merge_strategy == "any" and not successful_results:
                raise Exception("All parallel nodes failed")
            elif self.merge_strategy == "first":
                if successful_results:
                    new_state = successful_results[0]
            
            new_state.set(f"{self.name}_results", successful_results)
            new_state.set(f"{self.name}_errors", errors)
            
            self._mark_completed(new_state)
            return new_state
            
        except Exception as e:
            self._mark_failed(e)
            raise


@dataclass
class Edge:
    """图边 - 连接两个节点"""
    source: str
    target: str
    condition: Optional[Callable[[GraphState], bool]] = None
    label: str = ""


class AgentGraph:
    """Agent 编排图
    
    使用示例：
    ```python
    graph = AgentGraph("research_pipeline")
    
    # 添加节点
    graph.add_node(AgentNode("researcher", researcher_agent))
    graph.add_node(AgentNode("writer", writer_agent))
    graph.add_node(ConditionNode(
        "check_quality",
        lambda s: s.get("quality_score", 0) > 0.8,
        "publish", "revise"
    ))
    
    # 添加边
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "check_quality")
    graph.add_edge("check_quality", "publish")
    graph.add_edge("check_quality", "revise")
    graph.add_edge("revise", "writer")  # 循环
    
    # 执行
    result = await graph.run({"input": "研究 AI 趋势"})
    ```
    """
    
    def __init__(
        self,
        name: str,
        max_iterations: int = 100,
        checkpoint_dir: Optional[str] = None
    ):
        self.name = name
        self.max_iterations = max_iterations
        self.checkpoint_dir = checkpoint_dir
        
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: List[Edge] = []
        self._adjacency: Dict[str, List[Edge]] = {}
        self._reverse_adjacency: Dict[str, List[str]] = {}
        self._entry_nodes: Set[str] = set()
        self._exit_nodes: Set[str] = set()
        
        # 执行状态
        self._execution_id: Optional[str] = None
        self._state: Optional[GraphState] = None
        self._history: List[Dict[str, Any]] = []
        self._checkpoints: List[Dict[str, Any]] = []
    
    def add_node(self, node: GraphNode) -> 'AgentGraph':
        """添加节点"""
        if node.name in self._nodes:
            raise ValueError(f"Node '{node.name}' already exists")
        self._nodes[node.name] = node
        self._adjacency[node.name] = []
        self._reverse_adjacency[node.name] = []
        return self
    
    def add_edge(
        self,
        source: str,
        target: str,
        condition: Optional[Callable[[GraphState], bool]] = None,
        label: str = ""
    ) -> 'AgentGraph':
        """添加边"""
        if source not in self._nodes:
            raise ValueError(f"Source node '{source}' not found")
        if target not in self._nodes:
            raise ValueError(f"Target node '{target}' not found")
        
        edge = Edge(source=source, target=target, condition=condition, label=label)
        self._edges.append(edge)
        self._adjacency[source].append(edge)
        self._reverse_adjacency[target].append(source)
        
        # 更新入口/出口节点
        self._update_entry_exit()
        return self
    
    def add_conditional_edges(
        self,
        source: str,
        conditions: Dict[str, Callable[[GraphState], bool]],
        default: str
    ) -> 'AgentGraph':
        """添加条件边（用于 ConditionNode）"""
        # 这个方法用于从 ConditionNode 自动生成边
        # 实际路由在 ConditionNode.execute 中处理
        return self
    
    def _update_entry_exit(self):
        """更新入口和出口节点"""
        self._entry_nodes = {
            name for name in self._nodes
            if not self._reverse_adjacency[name]
        }
        self._exit_nodes = {
            name for name in self._nodes
            if not self._adjacency[name]
        }
    
    def _get_next_nodes(self, current_node: str, state: GraphState) -> List[str]:
        """获取下一步要执行的节点"""
        edges = self._adjacency.get(current_node, [])
        next_nodes = []
        
        for edge in edges:
            if edge.condition is None or edge.condition(state):
                next_nodes.append(edge.target)
        
        # 如果是 ConditionNode，检查状态中的 _next_node
        node = self._nodes.get(current_node)
        if isinstance(node, ConditionNode):
            next_node = state.get("_next_node")
            if next_node:
                return [next_node]
        
        return next_nodes
    
    async def run(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> GraphState:
        """执行图
        
        Args:
            initial_state: 初始状态数据
            config: 执行配置
            
        Returns:
            最终状态
        """
        self._execution_id = str(uuid.uuid4())[:8]
        self._state = GraphState(data=initial_state or {})
        self._history = []
        config = config or {}
        
        # 重置所有节点状态
        for node in self._nodes.values():
            node.status = NodeStatus.PENDING
            node.result = None
            node.error = None
        
        # 确定起始节点
        current_nodes = list(self._entry_nodes)
        if not current_nodes:
            # 没有明确入口，使用第一个添加的节点
            current_nodes = [list(self._nodes.keys())[0]]
        
        iteration = 0
        while current_nodes and iteration < self.max_iterations:
            iteration += 1
            
            next_nodes = []
            
            for node_name in current_nodes:
                node = self._nodes[node_name]
                
                # 检查依赖是否完成
                if not self._dependencies_met(node_name):
                    continue
                
                # 执行节点
                try:
                    self._state = await node.execute(self._state)
                    self._record_step(node_name, self._state)
                except Exception as e:
                    # 记录错误但继续（可配置）
                    self._record_step(node_name, self._state, error=str(e))
                    if config.get("fail_fast", True):
                        raise
                
                # 获取下一步节点
                next_for_node = self._get_next_nodes(node_name, self._state)
                next_nodes.extend(next_for_node)
            
            # 去重
            current_nodes = list(dict.fromkeys(next_nodes))
            
            # 检查是否所有出口节点完成
            if self._is_complete():
                break
            
            # 检查点保存
            if self.checkpoint_dir and iteration % config.get("checkpoint_interval", 10) == 0:
                self._save_checkpoint()
        
        if iteration >= self.max_iterations:
            print(f"⚠️ Graph {self.name} reached max iterations ({self.max_iterations})")
        
        return self._state
    
    def _dependencies_met(self, node_name: str) -> bool:
        """检查节点依赖是否满足"""
        predecessors = self._reverse_adjacency.get(node_name, [])
        if not predecessors:
            return True
        
        return all(
            self._nodes[p].status == NodeStatus.COMPLETED
            for p in predecessors
        )
    
    def _is_complete(self) -> bool:
        """检查是否所有出口节点完成"""
        if not self._exit_nodes:
            return all(
                n.status in (NodeStatus.COMPLETED, NodeStatus.SKIPPED, NodeStatus.FAILED)
                for n in self._nodes.values()
            )
        return all(
            self._nodes[n].status in (NodeStatus.COMPLETED, NodeStatus.SKIPPED)
            for n in self._exit_nodes
        )
    
    def _record_step(self, node_name: str, state: GraphState, error: str = None):
        """记录执行步骤"""
        node = self._nodes[node_name]
        self._history.append({
            "step": len(self._history) + 1,
            "node": node_name,
            "status": node.status.value,
            "duration_ms": node.duration_ms,
            "error": error,
            "state_keys": list(state.data.keys())
        })
    
    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint = {
            "execution_id": self._execution_id,
            "timestamp": datetime.now().isoformat(),
            "state": self._state.data.copy(),
            "node_statuses": {
                name: node.status.value for name, node in self._nodes.items()
            }
        }
        self._checkpoints.append(checkpoint)
        
        if self.checkpoint_dir:
            import json
            import os
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            path = os.path.join(self.checkpoint_dir, f"{self.name}_{self._execution_id}_checkpoint.json")
            with open(path, 'w') as f:
                json.dump(checkpoint, f, indent=2)

    # ==================== 循环检测 ====================

    def detect_cycles(self) -> List[List[str]]:
        """检测图中的所有环路
        
        使用 Tarjan 算法 (DFS + 栈) 查找强连通分量
        
        Returns:
            环路列表，每个环路是节点名称列表
        """
        index = 0
        stack = []
        on_stack = set()
        indices = {}
        lowlinks = {}
        cycles = []
        
        def strongconnect(node_name: str):
            nonlocal index
            indices[node_name] = index
            lowlinks[node_name] = index
            index += 1
            stack.append(node_name)
            on_stack.add(node_name)
            
            # 遍历所有出边
            for edge in self._adjacency.get(node_name, []):
                target = edge.target
                if target not in indices:
                    strongconnect(target)
                    lowlinks[node_name] = min(lowlinks[node_name], lowlinks[target])
                elif target in on_stack:
                    lowlinks[node_name] = min(lowlinks[node_name], indices[target])
            
            # 如果是强连通分量的根节点
            if lowlinks[node_name] == indices[node_name]:
                # 弹出栈直到找到当前节点
                scc = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    scc.append(w)
                    if w == node_name:
                        break
                # 如果 SCC 大小 > 1，或者有自环，则是环路
                if len(scc) > 1:
                    cycles.append(scc)
                elif len(scc) == 1:
                    # 检查自环
                    node_name = scc[0]
                    for edge in self._adjacency.get(node_name, []):
                        if edge.target == node_name:
                            cycles.append([node_name])
                            break
        
        for node_name in self._nodes:
            if node_name not in indices:
                strongconnect(node_name)
        
        return cycles

    def has_cycles(self) -> bool:
        """快速检查是否有环路"""
        return len(self.detect_cycles()) > 0

    def validate_dag(self) -> tuple[bool, List[List[str]]]:
        """验证是否为有向无环图 (DAG)
        
        Returns:
            (是否为DAG, 环路列表)
        """
        cycles = self.detect_cycles()
        return len(cycles) == 0, cycles

    # ==================== 可视化导出 ====================

    def to_mermaid(self, direction: str = "TD", show_status: bool = False) -> str:
        """导出为 Mermaid 流程图格式
        
        Args:
            direction: 流向 (TD=上下, LR=左右, RL=右左, BT=下上)
            show_status: 是否显示节点状态 (需先执行 run())
            
        Returns:
            Mermaid 格式字符串
        """
        lines = [f"graph {direction}", "    %% Nodes"]
        
        # 定义节点样式
        node_styles = {}
        for name, node in self._nodes.items():
            node_type = type(node).__name__
            label = f"{name}\\n({node_type})"
            
            # 根据节点类型选择形状
            if isinstance(node, ConditionNode):
                lines.append(f'    {name}["{label}"]:::condition')
            elif isinstance(node, ParallelNode):
                lines.append(f'    {name}["{label}"]:::parallel')
            elif isinstance(node, AgentNode):
                lines.append(f'    {name}["{label}"]:::agent')
            elif isinstance(node, FunctionNode):
                lines.append(f'    {name}["{label}"]:::function')
            else:
                lines.append(f'    {name}["{label}"]')
            
            # 如果有执行状态，记录用于后续渲染
            if show_status and hasattr(self, '_history'):
                node_styles[name] = node.status.value
        
        lines.append("    %% Edges")
        for edge in self._edges:
            label_part = f"|{edge.label}|" if edge.label else ""
            condition_part = ""
            if edge.condition:
                condition_part = " -.-> "
            else:
                condition_part = " --> "
            lines.append(f"    {edge.source}{condition_part}{label_part}{edge.target}")
        
        # 添加样式定义
        lines.append("")
        lines.append("    classDef agent fill:#e1f5fe,stroke:#01579b,stroke-width:2px;")
        lines.append("    classDef condition fill:#fff3e0,stroke:#e65100,stroke-width:2px,stroke-dasharray: 5, 5;")
        lines.append("    classDef parallel fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;")
        lines.append("    classDef function fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;")
        lines.append("    classDef completed fill:#c8e6c9,stroke:#2e7d32;")
        lines.append("    classDef failed fill:#ffcdd2,stroke:#c62828;")
        lines.append("    classDef running fill:#fff9c4,stroke:#fbc02d;")
        
        # 如果有状态，添加动态类
        if show_status:
            for name, status in node_styles.items():
                if status in ("completed", "failed", "running"):
                    lines.append(f"    class {name} {status};")
        
        return "\n".join(lines)

    def to_graphviz(self, direction: str = "TB", show_status: bool = False) -> str:
        """导出为 GraphViz DOT 格式
        
        Args:
            direction: 流向 (TB=上下, LR=左右, RL=右左, BT=下上)
            show_status: 是否显示节点状态
            
        Returns:
            DOT 格式字符串
        """
        lines = [
            f"digraph {self.name} {{",
            f"    rankdir={direction};",
            "    node [fontname=\"Arial\", fontsize=10];",
            "    edge [fontname=\"Arial\", fontsize=8];",
            ""
        ]
        
        # 节点定义
        for name, node in self._nodes.items():
            node_type = type(node).__name__
            label = f"{name}\\n({node_type})"
            
            # 根据类型选择形状和颜色
            if isinstance(node, ConditionNode):
                lines.append(f'    "{name}" [label="{label}", shape=diamond, style=filled, fillcolor="#fff3e0", color="#e65100"];')
            elif isinstance(node, ParallelNode):
                lines.append(f'    "{name}" [label="{label}", shape=box3d, style=filled, fillcolor="#f3e5f5", color="#4a148c"];')
            elif isinstance(node, AgentNode):
                lines.append(f'    "{name}" [label="{label}", shape=ellipse, style=filled, fillcolor="#e1f5fe", color="#01579b"];')
            elif isinstance(node, FunctionNode):
                lines.append(f'    "{name}" [label="{label}", shape=rect, style=filled, fillcolor="#e8f5e9", color="#1b5e20"];')
            else:
                lines.append(f'    "{name}" [label="{label}", shape=ellipse];')
        
        lines.append("")
        
        # 边定义
        for edge in self._edges:
            label = f' [label="{edge.label}"]' if edge.label else ""
            style = ' [style=dashed]' if edge.condition else ""
            lines.append(f'    "{edge.source}" -> "{edge.target}"{label}{style};')
        
        lines.append("}")
        return "\n".join(lines)

    def export_visualization(self, filepath: str, format: str = "mermaid") -> bool:
        """导出可视化到文件
        
        Args:
            filepath: 输出文件路径
            format: 格式 (mermaid, graphviz, json)
            
        Returns:
            是否成功
        """
        try:
            import os
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            
            if format == "mermaid":
                content = self.to_mermaid()
            elif format == "graphviz" or format == "dot":
                content = self.to_graphviz()
            elif format == "json":
                content = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"[ERROR] Export visualization failed: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """导出图结构为字典"""
        return {
            "name": self.name,
            "nodes": {
                name: {
                    "type": type(node).__name__,
                    "config": getattr(node, 'config', {}),
                }
                for name, node in self._nodes.items()
            },
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "label": edge.label,
                    "has_condition": edge.condition is not None
                }
                for edge in self._edges
            ],
            "entry_nodes": list(self._entry_nodes),
            "exit_nodes": list(self._exit_nodes),
            "cycles": self.detect_cycles(),
            "is_dag": not self.has_cycles()
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            "graph_name": self.name,
            "execution_id": self._execution_id,
            "total_steps": len(self._history),
            "nodes": {
                name: {
                    "status": node.status.value,
                    "duration_ms": node.duration_ms,
                    "error": str(node.error) if node.error else None
                }
                for name, node in self._nodes.items()
            },
            "history": self._history
        }


# 便捷函数：快速构建常见模式

def create_linear_graph(name: str, agents: List['Agent'], **kwargs) -> AgentGraph:
    """创建线性流水线图"""
    graph = AgentGraph(name, **kwargs)
    
    prev_node = None
    for i, agent in enumerate(agents):
        node = AgentNode(f"step_{i}", agent)
        graph.add_node(node)
        if prev_node:
            graph.add_edge(prev_node, node.name)
        prev_node = node.name
    
    return graph


def create_parallel_graph(name: str, agents: List['Agent'], **kwargs) -> AgentGraph:
    """创建并行执行图"""
    graph = AgentGraph(name, **kwargs)
    
    parallel_nodes = []
    for i, agent in enumerate(agents):
        node = AgentNode(f"parallel_{i}", agent)
        graph.add_node(node)
        parallel_nodes.append(node.name)
    
    # 添加一个汇聚节点
    from ..core.agent import Agent
    merge_agent = agents[0] if agents else None
    if merge_agent:
        merge_node = AgentNode("merge", merge_agent)
        graph.add_node(merge_node)
        for pn in parallel_nodes:
            graph.add_edge(pn, "merge")
    
    return graph


def create_conditional_graph(
    name: str,
    condition_agent: 'Agent',
    true_agent: 'Agent',
    false_agent: 'Agent',
    condition: Callable[[GraphState], bool],
    **kwargs
) -> AgentGraph:
    """创建条件分支图"""
    graph = AgentGraph(name, **kwargs)
    
    graph.add_node(AgentNode("condition", condition_agent))
    graph.add_node(ConditionNode("branch", condition, "true_branch", "false_branch"))
    graph.add_node(AgentNode("true_branch", true_agent))
    graph.add_node(AgentNode("false_branch", false_agent))
    
    graph.add_edge("condition", "branch")
    # ConditionNode 内部处理路由
    
    return graph