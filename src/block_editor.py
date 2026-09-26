"""
Qt Graphics View based block style editor.

Components
----------
Port              small circle on a node
Connection        bezier wire between an output and an input port
Node              a movable block with a title bar and rows of ports
ImageDisplayNode  a node that renders its input image live in-node
NodeScene         the background
NodeEditorView    wire-dragging, pan, zoom, delete, add-block menu
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, QPointF, QRectF

from .blocks import BlockDefinition, BlockRegistry, TYPE_COLORS

GItem = QtWidgets.QGraphicsItem
_Flag = GItem.GraphicsItemFlag
_Change = GItem.GraphicsItemChange

def _color(name: str):
    return QtGui.QColor(name)

def ndarray_to_qpixmap(arr):
    # Convert a BGR / grayscale uint8 numpy image to a QPixmap.
    if arr is None or not isinstance(arr, np.ndarray):
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        a = np.ascontiguousarray(arr)
        h, w = a.shape
        img = QtGui.QImage(a.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        a = np.ascontiguousarray(arr[:, :, ::-1])   # BGR -> RGB
        h, w, _ = a.shape
        img = QtGui.QImage(a.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
    else:
        return None
    return QtGui.QPixmap.fromImage(img.copy())

class Port(GItem):
    RADIUS = 6.0
    HIT_PAD = 4.0

    def __init__(self, node: "Node", name: str, is_output: bool, dtype: str):
        super().__init__(node)
        self.node = node
        self.name = name
        self.is_output = is_output
        self.dtype = dtype
        self.connections: List["Connection"] = []
        self._hover = False
        self.setZValue(3)
        self.setAcceptHoverEvents(True)

    def boundingRect(self):
        r = self.RADIUS + self.HIT_PAD
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        col = _color(TYPE_COLORS.get(self.dtype, TYPE_COLORS["any"]))
        painter.setBrush(QtGui.QBrush(col))
        pen = QtGui.QPen(_color("#f5f5f5" if self._hover else "#101418"))
        pen.setWidthF(1.6)
        painter.setPen(pen)
        r = self.RADIUS + (1.5 if self._hover else 0)
        painter.drawEllipse(QPointF(0, 0), r, r)

    def hoverEnterEvent(self, e):
        self._hover = True
        self.update()

    def hoverLeaveEvent(self, e):
        self._hover = False
        self.update()

    def can_connect_to(self, other: "Port"):
        if other is None or other is self:
            return False
        
        if other.node is self.node:
            return False
        
        if self.is_output == other.is_output:
            return False
        
        a, b = self.dtype, other.dtype
        
        if a != "any" and b != "any" and a != b:
            return False
        
        return True

class Connection(QtWidgets.QGraphicsPathItem):
    def __init__(self, start_port: Port, end_port: Optional[Port] = None):
        super().__init__()
        self.start_port = start_port
        self.end_port = end_port
        self._free_end: Optional[QPointF] = None
        self.setZValue(-1)
        self.setFlag(_Flag.ItemIsSelectable, True)
        self._apply_pen(False)

    def _apply_pen(self, selected: bool):
        col = _color("#ffd54f") if selected else _color(
            TYPE_COLORS.get(self.start_port.dtype, "#cfd8dc"))
        pen = QtGui.QPen(col)
        pen.setWidthF(2.6 if selected else 2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.setPen(pen)

    def set_free_end(self, scene_pos: QPointF):
        self._free_end = scene_pos

    def register(self):
        if self not in self.start_port.connections:
            self.start_port.connections.append(self)
        if self.end_port and self not in self.end_port.connections:
            self.end_port.connections.append(self)

    def remove(self):
        for p in (self.start_port, self.end_port):
            if p and self in p.connections:
                p.connections.remove(self)
        if self.scene():
            self.scene().removeItem(self)

    def update_path(self):
        p1 = self.start_port.scenePos()
        if self.end_port is not None:
            p2 = self.end_port.scenePos()
        elif self._free_end is not None:
            p2 = self._free_end
        else:
            p2 = p1
        out_pos, in_pos = (p1, p2) if self.start_port.is_output else (p2, p1)
        dx = max(30.0, abs(in_pos.x() - out_pos.x()) * 0.5)
        path = QtGui.QPainterPath(out_pos)
        path.cubicTo(out_pos.x() + dx, out_pos.y(),
                     in_pos.x() - dx, in_pos.y(),
                     in_pos.x(), in_pos.y())
        self.setPath(path)

    def paint(self, painter, option, widget=None):
        self._apply_pen(self.isSelected())
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(self.pen())
        painter.drawPath(self.path())

class Node(GItem):
    WIDTH = 168.0
    TITLE_H = 28.0
    ROW_H = 24.0
    PAD = 10.0
    is_display = False

    def __init__(self, definition: BlockDefinition, title: Optional[str] = None):
        super().__init__()
        self.definition = definition
        self.title = title or definition.name
        self.params: dict = {}
        for opt in definition.options:
            self.params.setdefault(opt.key, opt.default)
        self.inputs: List[Port] = []
        self.outputs: List[Port] = []
        self.setFlags(_Flag.ItemIsMovable | _Flag.ItemIsSelectable |
                      _Flag.ItemSendsGeometryChanges)
        self.setZValue(1)
        self._build_ports()
        self._layout_ports()

    def _build_ports(self):
        for name, dtype in self.definition.inputs:
            self.inputs.append(Port(self, name, False, dtype))
        for name, dtype in self.definition.outputs:
            self.outputs.append(Port(self, name, True, dtype))

    def _rows(self):
        return max(len(self.inputs), len(self.outputs), 1)

    def _has_info(self):
        return bool(self.definition.options)

    def height(self):
        extra = self.ROW_H if self._has_info() else 0.0
        return self.TITLE_H + self._rows() * self.ROW_H + extra + self.PAD

    def _layout_ports(self):
        y0 = self.TITLE_H + self.ROW_H / 2
        for i, p in enumerate(self.inputs):
            p.setPos(0.0, y0 + i * self.ROW_H)
        for i, p in enumerate(self.outputs):
            p.setPos(self.WIDTH, y0 + i * self.ROW_H)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.WIDTH, self.height())

    def _paint_body(self, painter):
        rect = QRectF(0, 0, self.WIDTH, self.height())
        radius = 8.0
        body = QtGui.QPainterPath()
        body.addRoundedRect(rect, radius, radius)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_color("#2b2f36"))
        painter.drawPath(body)

        title_rect = QRectF(0, 0, self.WIDTH, self.TITLE_H)
        painter.setClipPath(body)
        painter.setBrush(_color(self.definition.color))
        painter.drawRect(title_rect)
        painter.setClipping(False)

        painter.setPen(_color("#ffffff"))
        f = painter.font()
        f.setBold(True)
        f.setPointSizeF(9.5)
        painter.setFont(f)
        painter.drawText(title_rect.adjusted(10, 0, -10, 0),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         self.title)

        f.setBold(False)
        f.setPointSizeF(8.5)
        painter.setFont(f)
        painter.setPen(_color("#c7ccd1"))
        for p in self.inputs:
            y = p.pos().y()
            painter.drawText(QRectF(12, y - 10, self.WIDTH - 24, 20),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             p.name)
        for p in self.outputs:
            y = p.pos().y()
            painter.drawText(QRectF(12, y - 10, self.WIDTH - 24, 20),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                             p.name)

        if self._has_info():
            iy = self.TITLE_H + self._rows() * self.ROW_H + self.ROW_H / 2
            f.setBold(False)
            f.setPointSizeF(8.0)
            painter.setFont(f)
            painter.setPen(_color("#8a9299"))
            summary = "  ·  ".join(str(self.params.get(o.key, o.default))
                                   for o in self.definition.options)
            fm = QtGui.QFontMetrics(painter.font())
            summary = fm.elidedText(summary, Qt.TextElideMode.ElideRight,
                                    int(self.WIDTH - 20))
            painter.drawText(QRectF(10, iy - 9, self.WIDTH - 20, 18),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter,
                             summary)

        if self.isSelected():
            pen = QtGui.QPen(_color("#ffd54f"))
            pen.setWidthF(2.0)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(body)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self._paint_body(painter)

    def itemChange(self, change, value):
        if change == _Change.ItemPositionHasChanged:
            for p in self.inputs + self.outputs:
                for c in p.connections:
                    c.update_path()
        return super().itemChange(change, value)

class ImageDisplayNode(Node):
    DISP_W = 200.0
    DISP_H = 150.0
    is_display = True

    def __init__(self, definition: BlockDefinition, title: Optional[str] = None):
        self._pixmap: Optional[QtGui.QPixmap] = None
        super().__init__(definition, title)

    # widen so the preview fits comfortably
    WIDTH = 220.0

    def height(self):
        return self.TITLE_H + self._rows() * self.ROW_H + self.PAD + self.DISP_H + self.PAD

    def _disp_rect(self):
        top = self.TITLE_H + self._rows() * self.ROW_H + self.PAD
        x = (self.WIDTH - self.DISP_W) / 2
        return QRectF(x, top, self.DISP_W, self.DISP_H)

    def set_image(self, arr):
        self._pixmap = ndarray_to_qpixmap(arr)
        self.update()

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self._paint_body(painter)

        r = self._disp_rect()
        painter.setBrush(_color("#0e1013"))
        painter.setPen(_color("#3a3f45"))
        painter.drawRoundedRect(r, 5, 5)

        if self._pixmap is not None:
            inner = r.adjusted(4, 4, -4, -4)
            pm = self._pixmap.scaled(int(inner.width()), int(inner.height()),
                                     Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
            px = r.x() + (r.width() - pm.width()) / 2
            py = r.y() + (r.height() - pm.height()) / 2
            painter.drawPixmap(int(px), int(py), pm)
        else:
            painter.setPen(_color("#5a616a"))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, "no signal")

class ConstantNode(Node):
    def __init__(self, definition, title=None):
        super().__init__(definition, title)
        self.params.setdefault("value", 0.0)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self._paint_body(painter)
        y = self.TITLE_H + self.ROW_H / 2
        painter.setPen(_color("#ffd54f"))
        f = painter.font(); f.setBold(True); f.setPointSizeF(11.0)
        painter.setFont(f)
        painter.drawText(QRectF(12, y - 12, self.WIDTH - 24, 24),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         f"{self.params.get('value', 0.0):g}")

    def mouseDoubleClickEvent(self, event):
        views = self.scene().views() if self.scene() else []
        parent = views[0] if views else None
        cur = float(self.params.get("value", 0.0))
        val, ok = QtWidgets.QInputDialog.getDouble(
            parent, "Constant", "Value:", cur, -1e9, 1e9, 4)
        if ok:
            self.params["value"] = val
            self.update()

def make_node(definition: BlockDefinition):
    if definition.display:
        return ImageDisplayNode(definition)
    if definition.constant:
        return ConstantNode(definition)
    return Node(definition)

class NodeScene(QtWidgets.QGraphicsScene):
    GRID = 24

    def __init__(self):
        super().__init__()
        self.setSceneRect(-5000, -5000, 10000, 10000)

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, _color("#1a1d21"))
        left = int(rect.left()) - (int(rect.left()) % self.GRID)
        top = int(rect.top()) - (int(rect.top()) % self.GRID)

        fine = QtGui.QPen(_color("#23272c"))
        fine.setWidthF(1.0)
        coarse = QtGui.QPen(_color("#2c3137"))
        coarse.setWidthF(1.0)

        x = left
        while x < rect.right():
            painter.setPen(coarse if (x % (self.GRID * 5) == 0) else fine)
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
            x += self.GRID
        y = top
        while y < rect.bottom():
            painter.setPen(coarse if (y % (self.GRID * 5) == 0) else fine)
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)
            y += self.GRID

class NodeEditorView(QtWidgets.QGraphicsView):
    def __init__(self, scene: NodeScene, registry: BlockRegistry):
        super().__init__(scene)
        self.registry = registry
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._temp: Optional[Connection] = None
        self._drag_port: Optional[Port] = None
        self._panning = False
        self._pan_last = QtCore.QPoint()

    def _port_at(self, view_pos):
        for it in self.items(view_pos):
            if isinstance(it, Port):
                return it
        return None

    def add_block(self, name: str, scene_pos: QPointF):
        node = make_node(self.registry.get(name))
        node.setPos(scene_pos)
        self.scene().addItem(node)
        return node

    def _make_connection(self, a: Port, b: Port):
        out_port = a if a.is_output else b
        in_port = b if a.is_output else a
        for c in list(in_port.connections):
            c.remove()
        conn = Connection(out_port, in_port)
        conn.register()
        self.scene().addItem(conn)
        conn.update_path()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            port = self._port_at(e.pos())
            if port is not None:
                self._drag_port = port
                self._temp = Connection(port)
                self._temp.set_free_end(self.mapToScene(e.pos()))
                self._temp.update_path()
                self.scene().addItem(self._temp)
                return
        elif e.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_last = e.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._temp is not None:
            self._temp.set_free_end(self.mapToScene(e.pos()))
            self._temp.update_path()
            return
        if self._panning:
            delta = e.pos() - self._pan_last
            self._pan_last = e.pos()
            self.translate(delta.x(), delta.y())
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._temp is not None:
            target = self._port_at(e.pos())
            start = self._drag_port
            self._temp.remove()
            self._temp = None
            self._drag_port = None
            if target is not None and start.can_connect_to(target):
                self._make_connection(start, target)
            return
        if self._panning and e.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.unsetCursor()
            return
        super().mouseReleaseEvent(e)

    def wheelEvent(self, e):
        factor = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            for it in list(self.scene().selectedItems()):
                if isinstance(it, Connection):
                    it.remove()
                elif isinstance(it, Node):
                    self._delete_node(it)
            return
        super().keyPressEvent(e)

    def _delete_node(self, node: Node):
        for p in node.inputs + node.outputs:
            for c in list(p.connections):
                c.remove()
        self.scene().removeItem(node)

    def _node_at(self, view_pos):
        for it in self.items(view_pos):
            if isinstance(it, Node):
                return it
        return None

    def _show_node_menu(self, node, global_pos):
        menu = QtWidgets.QMenu(self)
        for opt in node.definition.options:
            sub = menu.addMenu(opt.label)
            group = QtGui.QActionGroup(sub)
            group.setExclusive(True)
            current = node.params.get(opt.key, opt.default)
            for value, display in opt.choices:
                act = sub.addAction(display)
                act.setCheckable(True)
                act.setChecked(value == current)
                act.setData(("set", opt.key, value))
                group.addAction(act)
        if node.definition.options:
            menu.addSeparator()
        act_del = menu.addAction("Delete")
        act_del.setData(("delete",))

        chosen = menu.exec(global_pos)
        if chosen is None:
            return
        data = chosen.data()
        if not data:
            return
        if data[0] == "delete":
            self._delete_node(node)
        elif data[0] == "set":
            _, key, value = data
            node.params[key] = value
            node.update()

    def contextMenuEvent(self, e):
        node = self._node_at(e.pos())
        if node is not None:
            self._show_node_menu(node, e.globalPos())
            return
        menu = QtWidgets.QMenu(self)
        for category, defs in self.registry.by_category().items():
            sub = menu.addMenu(category)
            for d in defs:
                act = sub.addAction(d.name)
                act.setData(d.name)
        chosen = menu.exec(e.globalPos())
        if chosen is not None:
            self.add_block(chosen.data(), self.mapToScene(e.pos()))