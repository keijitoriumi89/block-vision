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
    "lines":    "#ef5350",
    "scalar":   "#ffb74d",
    "any":      "#9e9e9e",
}

CATEGORY_COLORS = {
    "Source":     "#2e7d32",
    "Input":      "#f9a325",
    "Processing": "#1565c0",
    "Features":   "#6a1b9a",
    "Output":     "#c62828",
    "General":    "#455a64",
}

@dataclass
class ChoiceParam:
    """A categorical parameter, edited from the node's right-click menu."""
    key: str
    label: str
    choices: List[Tuple[str, str]]   # (value, display) pairs
    default: str


@dataclass
class BlockDefinition:
    name: str
    category: str = "General"
    inputs: List[PortSpec] = field(default_factory=list)
    outputs: List[PortSpec] = field(default_factory=list)
    params: List[dict] = field(default_factory=list)
    compute: Optional[Callable] = None
    display: bool = False
    constant: bool = False
    options: List[ChoiceParam] = field(default_factory=list)

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

def _constant(inputs, params, ctx):
    return {"value": params.get("value", 0.0)}

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

_THRESH = {"binary": cv2.THRESH_BINARY, "binary_inv": cv2.THRESH_BINARY_INV,
           "trunc": cv2.THRESH_TRUNC, "tozero": cv2.THRESH_TOZERO,
           "tozero_inv": cv2.THRESH_TOZERO_INV}


def _threshold(inputs, params, ctx):
    g = _as_gray(inputs.get("gray"))
    if g is None:
        return {"mask": None}
    t = inputs.get("thresh")
    t = 127.0 if t is None else float(t)
    ttype = params.get("type", "binary")
    if ttype == "otsu":                      # Otsu picks the threshold itself
        _, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(g, t, 255, _THRESH.get(ttype, cv2.THRESH_BINARY))
    return {"mask": mask}

_INTERP = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR,
           "area": cv2.INTER_AREA, "cubic": cv2.INTER_CUBIC,
           "lanczos4": cv2.INTER_LANCZOS4}


def _resize(inputs, params, ctx):
    img = inputs.get("image")
    if img is None:
        return {"image": None}
    s = inputs.get("scale")
    s = 0.5 if s is None else float(s)
    s = max(0.05, s)
    interp = _INTERP.get(params.get("interp", "area"), cv2.INTER_AREA)
    return {"image": cv2.resize(img, None, fx=s, fy=s, interpolation=interp)}

def _canny(inputs, params, ctx):
    g = _as_gray(inputs.get("gray"))
    if g is None:
        return {"edges": None}
    lo = inputs.get("low")
    hi = inputs.get("high")
    lo = 50.0 if lo is None else float(lo)
    hi = 150.0 if hi is None else float(hi)
    return {"edges": cv2.Canny(g, lo, hi)}

_RETR = {"external": cv2.RETR_EXTERNAL, "list": cv2.RETR_LIST,
         "tree": cv2.RETR_TREE, "ccomp": cv2.RETR_CCOMP}
_APPROX = {"simple": cv2.CHAIN_APPROX_SIMPLE, "none": cv2.CHAIN_APPROX_NONE}


def _find_contours(inputs, params, ctx):
    m = _as_gray(inputs.get("mask"))
    if m is None:
        return {"image": None, "contours": None, "count": 0.0}
    mask = m.astype(np.uint8)

    mode = _RETR.get(params.get("mode", "external"), cv2.RETR_EXTERNAL)
    method = _APPROX.get(params.get("method", "simple"), cv2.CHAIN_APPROX_SIMPLE)
    cnts, _ = cv2.findContours(mask, mode, method)

    min_area = inputs.get("min_area")
    if min_area is not None and float(min_area) > 0:
        cnts = [c for c in cnts if cv2.contourArea(c) >= float(min_area)]

    base = inputs.get("image")               # draw on the image if connected,
    if base is not None:                     # otherwise on the mask itself
        canvas = base.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    else:
        canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    color = (0, 255, 0)
    style = params.get("draw", "outline")
    if style == "outline":
        cv2.drawContours(canvas, cnts, -1, color, 2)
    elif style == "filled":
        cv2.drawContours(canvas, cnts, -1, color, cv2.FILLED)
    elif style == "bbox":
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
    elif style == "hull":
        cv2.drawContours(canvas, [cv2.convexHull(c) for c in cnts], -1, color, 2)

    return {"image": canvas, "contours": cnts, "count": float(len(cnts))}


def _hough_lines(inputs, params, ctx):
    edges = _as_gray(inputs.get("edges"))
    if edges is None:
        return {"image": None, "lines": None, "count": 0.0}
    edges = edges.astype(np.uint8)

    thresh = inputs.get("threshold")
    thresh = 80 if thresh is None else max(1, int(thresh))

    base = inputs.get("image")               # draw over the image if connected,
    if base is not None:                     # otherwise over the edge map
        canvas = base.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    else:
        canvas = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    color = (0, 0, 255)
    if params.get("method", "prob") == "prob":       # HoughLinesP -> segments
        min_len = inputs.get("min_length")
        min_len = 50.0 if min_len is None else float(min_len)
        max_gap = inputs.get("max_gap")
        max_gap = 10.0 if max_gap is None else float(max_gap)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, thresh,
                                minLineLength=min_len, maxLineGap=max_gap)
        segs = [] if lines is None else lines.reshape(-1, 4)
        for x1, y1, x2, y2 in segs:
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        n = len(segs)
    else:                                            # HoughLines -> infinite
        lines = cv2.HoughLines(edges, 1, np.pi / 180, thresh)
        rts = [] if lines is None else lines.reshape(-1, 2)
        for rho, theta in rts:
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            p1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            p2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(canvas, p1, p2, color, 2)
        n = len(rts)

    return {"image": canvas, "lines": lines, "count": float(n)}

def build_default_registry():
    reg = BlockRegistry()

    reg.register(BlockDefinition(
        "Constant", "Input", inputs=[], outputs=[("value", "scalar")],
        compute=_constant, constant=True))

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
        outputs=[("mask", "mask")], compute=_threshold,
        options=[ChoiceParam("type", "Type", [
            ("binary", "Binary"), ("binary_inv", "Binary inv"),
            ("otsu", "Otsu"), ("trunc", "Truncate"),
            ("tozero", "To zero"), ("tozero_inv", "To zero inv")], "binary")]))
    reg.register(BlockDefinition(
        "Resize", "Processing",
        inputs=[("image", "any"), ("scale", "scalar")],
        outputs=[("image", "any")], compute=_resize,
        options=[ChoiceParam("interp", "Interpolation", [
            ("nearest", "Nearest"), ("linear", "Linear"), ("area", "Area"),
            ("cubic", "Cubic"), ("lanczos4", "Lanczos4")], "area")]))

    reg.register(BlockDefinition(
        "Canny Edges", "Features",
        inputs=[("gray", "gray"), ("low", "scalar"), ("high", "scalar")],
        outputs=[("edges", "mask")], compute=_canny))
    reg.register(BlockDefinition(
        "Find Contours", "Features",
        inputs=[("mask", "mask"), ("image", "image"), ("min_area", "scalar")],
        outputs=[("image", "any"), ("contours", "contours"),
                 ("count", "scalar")],
        compute=_find_contours,
        options=[
            ChoiceParam("mode", "Retrieval", [
                ("external", "External"), ("list", "List"),
                ("tree", "Tree"), ("ccomp", "CComp")], "external"),
            ChoiceParam("method", "Approx", [
                ("simple", "Simple"), ("none", "None")], "simple"),
            ChoiceParam("draw", "Draw", [
                ("outline", "Outline"), ("filled", "Filled"),
                ("bbox", "Boxes"), ("hull", "Hulls")], "outline"),
        ]))
    reg.register(BlockDefinition(
        "Hough Lines", "Features",
        inputs=[("edges", "mask"), ("image", "image"),
                ("threshold", "scalar"), ("min_length", "scalar"),
                ("max_gap", "scalar")],
        outputs=[("image", "any"), ("lines", "lines"), ("count", "scalar")],
        compute=_hough_lines,
        options=[ChoiceParam("method", "Method", [
            ("prob", "Probabilistic"), ("standard", "Standard")], "prob")]))

    reg.register(BlockDefinition(
        "Viewer", "Output", inputs=[("image", "any")], outputs=[],
        compute=None, display=True))

    return reg