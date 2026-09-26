from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

PortSpec = Tuple[str, str]

TYPE_COLORS = {
    "image":    "#4fc3f7",
    "gray":     "#b0bec5",
    "mask":     "#ba68c8",
    "contours": "#81c784",
    "scalar":   "#ffb74d",
    "any":      "#9e9e9e",
}

CATEGORY_COLORS = {
    "Source":     "#2e7d32",
    "Processing": "#1565c0",
    "Features":   "#6a1b9a",
    "Output":     "#c62828",
    "General":    "#455a64",
}

@dataclass
class BlockDefinition:
    name: str
    category: str = "General"
    inputs: List[PortSpec] = field(default_factory=list)
    outputs: List[PortSpec] = field(default_factory=list)
    params: List[dict] = field(default_factory=list)
    compute: Optional[Callable] = None
    display: bool = False

    @property
    def color(self):
        return CATEGORY_COLORS.get(self.category, CATEGORY_COLORS["General"])

class BlockRegistry:
    def __init__(self):
        self._defs: dict[str, BlockDefinition] = {}

    def register(self, definition: BlockDefinition):
        if definition.name in self._defs:
            raise ValueError(f"Block '{definition.name}' already registered")
        self._defs[definition.name] = definition
        return definition

    def get(self, name: str):
        return self._defs[name]

    def all(self):
        return list(self._defs.values())

    def by_category(self) -> dict[str, List[BlockDefinition]]:
        groups: dict[str, List[BlockDefinition]] = {}
        for d in self._defs.values():
            groups.setdefault(d.category, []).append(d)
        return groups

def _as_gray(img):
    if img is None:
        return None
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _camera(inputs, params, ctx):
    return {"frame": ctx.frame}

def _image_file(inputs, params, ctx):
    path = params.get("path")
    return {"image": cv2.imread(path) if path else None}

def _grayscale(inputs, params, ctx):
    return {"gray": _as_gray(inputs.get("image"))}

def _gaussian_blur(inputs, params, ctx):
    img = inputs.get("image")
    if img is None:
        return {"image": None}
    k = int(inputs.get("ksize") or params.get("ksize", 5))
    if k % 2 == 0:
        k += 1
    k = max(1, k)
    return {"image": cv2.GaussianBlur(img, (k, k), 0)}

def _threshold(inputs, params, ctx):
    g = _as_gray(inputs.get("gray"))
    if g is None:
        return {"mask": None}
    t = inputs.get("thresh")
    t = 127.0 if t is None else float(t)
    _, mask = cv2.threshold(g, t, 255, cv2.THRESH_BINARY)
    return {"mask": mask}

def _resize(inputs, params, ctx):
    img = inputs.get("image")
    if img is None:
        return {"image": None}
    s = inputs.get("scale")
    s = 0.5 if s is None else float(s)
    s = max(0.05, s)
    return {"image": cv2.resize(img, None, fx=s, fy=s,
                                interpolation=cv2.INTER_AREA)}

def _canny(inputs, params, ctx):
    g = _as_gray(inputs.get("gray"))
    if g is None:
        return {"edges": None}
    lo = inputs.get("low")
    hi = inputs.get("high")
    lo = 50.0 if lo is None else float(lo)
    hi = 150.0 if hi is None else float(hi)
    return {"edges": cv2.Canny(g, lo, hi)}

def _find_contours(inputs, params, ctx):
    m = _as_gray(inputs.get("mask"))
    if m is None:
        return {"contours": None}
    cnts, _ = cv2.findContours(m.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return {"contours": cnts}

def build_default_registry():
    reg = BlockRegistry()

    reg.register(BlockDefinition(
        "Camera", "Source", inputs=[], outputs=[("frame", "image")],
        compute=_camera))
    reg.register(BlockDefinition(
        "Image File", "Source", inputs=[], outputs=[("image", "image")],
        compute=_image_file))

    reg.register(BlockDefinition(
        "Grayscale", "Processing",
        inputs=[("image", "image")], outputs=[("gray", "gray")],
        compute=_grayscale))
    reg.register(BlockDefinition(
        "Gaussian Blur", "Processing",
        inputs=[("image", "any"), ("ksize", "scalar")],
        outputs=[("image", "any")], compute=_gaussian_blur))
    reg.register(BlockDefinition(
        "Threshold", "Processing",
        inputs=[("gray", "gray"), ("thresh", "scalar")],
        outputs=[("mask", "mask")], compute=_threshold))
    reg.register(BlockDefinition(
        "Resize", "Processing",
        inputs=[("image", "any"), ("scale", "scalar")],
        outputs=[("image", "any")], compute=_resize))

    reg.register(BlockDefinition(
        "Canny Edges", "Features",
        inputs=[("gray", "gray"), ("low", "scalar"), ("high", "scalar")],
        outputs=[("edges", "mask")], compute=_canny))
    reg.register(BlockDefinition(
        "Find Contours", "Features",
        inputs=[("mask", "mask")], outputs=[("contours", "contours")],
        compute=_find_contours))

    reg.register(BlockDefinition(
        "Viewer", "Output", inputs=[("image", "any")], outputs=[],
        compute=None, display=True))

    return reg