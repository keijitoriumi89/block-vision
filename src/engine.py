from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
from PySide6 import QtCore

from .block_editor import Node, Connection

@dataclass
class ExecutionContext:
    frame: Optional[object] = None
    frame_index: int = 0


def _topological_order(nodes, connections):
    indeg = {n: 0 for n in nodes}
    adj = {n: [] for n in nodes}
    for c in connections:
        if c.start_port is None or c.end_port is None:
            continue
        a, b = c.start_port.node, c.end_port.node
        if a in adj and b in indeg:
            adj[a].append(b)
            indeg[b] += 1
    q = deque(n for n in nodes if indeg[n] == 0)
    order = []
    while q:
        n = q.popleft()
        order.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    # any nodes left are part of a cycle; append them so they still run once
    for n in nodes:
        if n not in order:
            order.append(n)
    return order


class GraphExecutor:
    """Stateless-per-call evaluator over the current scene contents."""

    def step(self, scene, ctx: ExecutionContext) -> dict:
        nodes = [i for i in scene.items() if isinstance(i, Node)]
        conns = [i for i in scene.items() if isinstance(i, Connection)]

        # input port -> upstream output port
        src = {c.end_port: c.start_port
               for c in conns if c.start_port and c.end_port}

        cache: dict = {}   # output Port -> value
        for node in _topological_order(nodes, conns):
            inputs = {p.name: cache.get(src.get(p)) for p in node.inputs}
            outputs = {}
            comp = node.definition.compute
            if comp is not None:
                try:
                    outputs = comp(inputs, node.params, ctx) or {}
                except Exception as ex:                       # keep the loop alive
                    print(f"[block error] {node.title}: {ex}")
            for p in node.outputs:
                cache[p] = outputs.get(p.name)

        # feed display nodes
        for node in nodes:
            if getattr(node, "is_display", False) and node.inputs:
                node.set_image(cache.get(src.get(node.inputs[0])))

        return cache


class PipelineRunner(QtCore.QObject):
    """Owns the webcam + a QTimer; drives one executor step per frame."""

    error = QtCore.Signal(str)
    started = QtCore.Signal()
    stopped = QtCore.Signal()

    def __init__(self, scene, executor: GraphExecutor,
                 device: int = 0, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.executor = executor
        self.device = device
        self.cap: Optional[cv2.VideoCapture] = None
        self.ctx = ExecutionContext()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)

    @property
    def running(self) -> bool:
        return self.timer.isActive()

    def start(self, fps: int = 30) -> bool:
        if self.running:
            return True
        cap = cv2.VideoCapture(self.device)
        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            self.error.emit(
                f"Could not open webcam (device {self.device}). "
                "Check it isn't in use by another app.")
            return False
        self.cap = cap
        self.ctx.frame_index = 0
        self.timer.start(max(1, int(1000 / fps)))
        self.started.emit()
        return True

    def stop(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.stopped.emit()

    def _tick(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        self.ctx.frame = frame
        self.ctx.frame_index += 1
        self.executor.step(self.scene, self.ctx)