"""
Entry point for the block-based perception pipeline.
"""

from __future__ import annotations
import sys

from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, QPointF

from src.blocks import build_default_registry
from src.block_editor import NodeScene, NodeEditorView
from src.engine import GraphExecutor, PipelineRunner

WEBCAM_DEVICE = 0

class PalettePanel(QtWidgets.QWidget):
    def __init__(self, view: NodeEditorView):
        super().__init__()
        self.view = view
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(12)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.tree)

        self._populate()

    def _populate(self):
        for category, defs in self.view.registry.by_category().items():
            top = QtWidgets.QTreeWidgetItem([category])
            top.setFlags(Qt.ItemFlag.ItemIsEnabled)
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            self.tree.addTopLevelItem(top)
            for d in defs:
                child = QtWidgets.QTreeWidgetItem([d.name])
                child.setData(0, Qt.ItemDataRole.UserRole, d.name)
                top.addChild(child)
            top.setExpanded(True)

    def _on_double_click(self, item, _col):
        name = item.data(0, Qt.ItemDataRole.UserRole)
        if not name:
            return
        center = self.view.mapToScene(self.view.viewport().rect().center())
        self.view.add_block(name, center)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Perception Pipeline — Block Editor")
        self.resize(1200, 760)

        self.registry = build_default_registry()
        self.scene = NodeScene()
        self.view = NodeEditorView(self.scene, self.registry)
        self.setCentralWidget(self.view)

        # execution
        self.executor = GraphExecutor()
        self.runner = PipelineRunner(self.scene, self.executor,
                                     source=WEBCAM_DEVICE)
        self.runner.error.connect(self._on_runner_error)
        self.runner.started.connect(
            lambda: self.statusBar().showMessage("Running — webcam live"))
        self.runner.stopped.connect(
            lambda: self.statusBar().showMessage("Stopped"))

        # palette dock
        dock = QtWidgets.QDockWidget("Blocks", self)
        dock.setWidget(PalettePanel(self.view))
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        self._build_toolbar()
        self.statusBar().showMessage("Ready — press Play to start the webcam")
        self._seed_demo()

    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)

        self.play_act = QtGui.QAction("Play", self)
        self.play_act.setCheckable(True)
        self.play_act.setShortcut("Space")
        self.play_act.toggled.connect(self._toggle_play)
        tb.addAction(self.play_act)
        tb.addAction("Open Video…", self._open_video)
        tb.addAction("Use Webcam", self._use_webcam)

        tb.addSeparator()
        tb.addAction("Fit View", self._fit_view)
        tb.addAction("Reset Zoom", lambda: self.view.resetTransform())
        tb.addAction("Clear", self._clear)

    def _toggle_play(self, on: bool):
        if on:
            if self.runner.start():
                self.play_act.setText("Stop")
            else:
                self.play_act.blockSignals(True)
                self.play_act.setChecked(False)
                self.play_act.blockSignals(False)
        else:
            self.runner.stop()
            self.play_act.setText("Play")

    def _on_runner_error(self, msg: str):
        QtWidgets.QMessageBox.warning(self, "Webcam", msg)
        self.statusBar().showMessage(msg)

    def _seed_demo(self):
        cam = self.view.add_block("Camera", QPointF(-320, -30))
        gray = self.view.add_block("Grayscale", QPointF(-90, -50))
        canny = self.view.add_block("Canny Edges", QPointF(150, -60))
        viewer = self.view.add_block("Viewer", QPointF(410, -40))
        c_low = self.view.add_block("Constant", QPointF(-120, 150))
        c_high = self.view.add_block("Constant", QPointF(-120, 230))
        c_low.params["value"] = 50.0
        c_high.params["value"] = 150.0
        self.view._make_connection(cam.outputs[0], gray.inputs[0])
        self.view._make_connection(gray.outputs[0], canny.inputs[0])
        self.view._make_connection(c_low.outputs[0], canny.inputs[1])
        self.view._make_connection(c_high.outputs[0], canny.inputs[2])
        self.view._make_connection(canny.outputs[0], viewer.inputs[0])
        QtCore.QTimer.singleShot(0, self._fit_view)

    def _fit_view(self):
        items = self.scene.itemsBoundingRect()
        if items.isValid():
            self.view.fitInView(items.adjusted(-60, -60, 60, 60),
                                Qt.AspectRatioMode.KeepAspectRatio)

    def _clear(self):
        was_running = self.runner.running
        self.runner.stop()
        self.scene.clear()
        if was_running:
            self.play_act.setChecked(False)

    def _set_source_and_play(self, source, label):
        self.runner.set_source(source)          # restarts if already running
        self.statusBar().showMessage(f"Source: {label}")
        if not self.play_act.isChecked():
            self.play_act.setChecked(True)      # triggers start

    def _open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if path:
            self._set_source_and_play(path, path)

    def _use_webcam(self):
        self._set_source_and_play(WEBCAM_DEVICE, "webcam")

    def closeEvent(self, e):
        self.runner.stop()
        super().closeEvent(e)

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = app.palette()
    pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#22262b"))
    pal.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#1a1d21"))
    pal.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#e6e9ec"))
    pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#e6e9ec"))
    pal.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#2b3036"))
    pal.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#e6e9ec"))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()