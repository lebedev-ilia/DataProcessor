from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class GraphNode:
    component_name: str
    owner_processor: str
    depends_on_components: Tuple[str, ...] = ()
    soft_dependencies: Tuple[str, ...] = ()
    wait_on_checkpoints: Tuple[str, ...] = ()


class ComponentGraph:
    def __init__(self, *, version: str, stage: str, nodes: List[GraphNode]) -> None:
        self.version = str(version)
        self.stage = str(stage)
        self.nodes = list(nodes)
        self.by_name: Dict[str, GraphNode] = {n.component_name: n for n in nodes}

    @staticmethod
    def from_yaml_dict(payload: Dict[str, Any], *, stage: str) -> "ComponentGraph":
        if not isinstance(payload, dict):
            raise ValueError("component_graph.yaml: root must be a dict")
        version = str(payload.get("version") or "unknown")
        stages = payload.get("stages") or {}
        if not isinstance(stages, dict):
            raise ValueError("component_graph.yaml: stages must be a dict")
        st = stages.get(stage)
        if not isinstance(st, dict):
            raise ValueError(f"component_graph.yaml: stage '{stage}' not found")
        nodes_raw = st.get("nodes") or []
        if not isinstance(nodes_raw, list):
            raise ValueError(f"component_graph.yaml: stages.{stage}.nodes must be a list")

        nodes: List[GraphNode] = []
        for i, n in enumerate(nodes_raw):
            if not isinstance(n, dict):
                raise ValueError(f"component_graph.yaml: node[{i}] must be a dict")
            name = n.get("component_name")
            owner = n.get("owner_processor")
            if not isinstance(name, str) or not name:
                raise ValueError(f"component_graph.yaml: node[{i}] missing component_name")
            if not isinstance(owner, str) or not owner:
                raise ValueError(f"component_graph.yaml: node[{i}] missing owner_processor for {name}")

            def _lst(key: str) -> Tuple[str, ...]:
                v = n.get(key) or []
                if not isinstance(v, list):
                    raise ValueError(f"component_graph.yaml: node[{i}].{key} must be a list")
                out = []
                for x in v:
                    if not isinstance(x, str) or not x:
                        raise ValueError(f"component_graph.yaml: node[{i}].{key} must contain strings")
                    out.append(x)
                return tuple(out)

            nodes.append(
                GraphNode(
                    component_name=name,
                    owner_processor=owner,
                    depends_on_components=_lst("depends_on_components"),
                    soft_dependencies=_lst("soft_dependencies"),
                    wait_on_checkpoints=_lst("wait_on_checkpoints"),
                )
            )

        g = ComponentGraph(version=version, stage=stage, nodes=nodes)
        g.validate()
        return g

    def validate(self) -> None:
        # Unique names
        names = [n.component_name for n in self.nodes]
        if len(set(names)) != len(names):
            dupes = sorted({x for x in names if names.count(x) > 1})
            raise ValueError(f"component_graph: duplicate component_name(s): {dupes}")

        # All hard deps exist in same stage (MVP)
        missing: List[Tuple[str, str]] = []
        for n in self.nodes:
            for dep in n.depends_on_components:
                if dep not in self.by_name:
                    missing.append((n.component_name, dep))
        if missing:
            msg = ", ".join([f"{a} depends_on {b}" for a, b in missing[:20]])
            raise ValueError(f"component_graph: missing hard dependencies: {msg}")

        # Acyclic check
        visiting: Set[str] = set()
        visited: Set[str] = set()

        def dfs(x: str) -> None:
            if x in visited:
                return
            if x in visiting:
                raise ValueError(f"component_graph: cycle detected at {x}")
            visiting.add(x)
            node = self.by_name.get(x)
            if node:
                for dep in node.depends_on_components:
                    dfs(dep)
            visiting.remove(x)
            visited.add(x)

        for name in names:
            dfs(name)

    def topo_order(self, subset: Optional[Iterable[str]] = None) -> List[str]:
        """
        Return deterministic topo order.
        If subset is provided, returns order for subset + their transitive hard deps.
        """
        wanted: Set[str] = set(self.by_name.keys()) if subset is None else set(subset)

        # Expand by transitive deps
        def add_deps(x: str) -> None:
            if x not in self.by_name:
                return
            for d in self.by_name[x].depends_on_components:
                if d not in wanted:
                    wanted.add(d)
                    add_deps(d)

        for x in list(wanted):
            add_deps(x)

        # Kahn topo sort with stable ordering (by name)
        indeg: Dict[str, int] = {n: 0 for n in wanted}
        children: Dict[str, List[str]] = {n: [] for n in wanted}
        for n in wanted:
            node = self.by_name.get(n)
            if not node:
                continue
            for d in node.depends_on_components:
                if d in wanted:
                    indeg[n] += 1
                    children[d].append(n)

        ready = sorted([n for n, deg in indeg.items() if deg == 0])
        out: List[str] = []
        while ready:
            x = ready.pop(0)
            out.append(x)
            for ch in sorted(children.get(x, [])):
                indeg[ch] -= 1
                if indeg[ch] == 0:
                    # insert keeping ready sorted
                    ready.append(ch)
                    ready.sort()

        if len(out) != len(wanted):
            raise ValueError("component_graph: cycle (topo sort incomplete)")
        return out


