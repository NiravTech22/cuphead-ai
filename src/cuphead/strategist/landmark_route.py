"""A visual breadcrumb graph: route between verified landmarks, then re-localize.

A route is a graph rather than a blind timed script. Short edge actions can be
reused after a death or an unexpected menu because navigation starts from the
landmark currently visible. Failed transitions never advance the route.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteEdge:
    source: str
    target: str
    action: str


class LandmarkRoute:
    def __init__(self, edges: list[RouteEdge], goal: str):
        if not goal or any(not e.source or not e.target or not e.action for e in edges):
            raise ValueError("route labels and actions must be nonempty")
        self.edges, self.goal = list(edges), goal
        self.confirmed: dict[tuple[str, str, str], int] = {}

    def next_edge(self, current: str) -> RouteEdge | None:
        if current == self.goal:
            return None
        queue = deque([(current, None)])
        visited = {current}
        while queue:
            node, first = queue.popleft()
            for edge in self.edges:
                if edge.source != node or edge.target in visited:
                    continue
                head = first or edge
                if edge.target == self.goal:
                    return head
                visited.add(edge.target)
                queue.append((edge.target, head))
        return None

    def observe(self, before: str, action: str, after: str) -> bool:
        """Only observed changes confirm a breadcrumb; elapsed time proves nothing."""
        if before == after:
            return False
        key = (before, action, after)
        self.confirmed[key] = self.confirmed.get(key, 0) + 1
        edge = RouteEdge(before, after, action)
        if edge not in self.edges:
            self.edges.append(edge)
        return True
