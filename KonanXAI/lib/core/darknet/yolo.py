from .api import *
import math

class BBox:
    def __init__(self, in_w, in_h, cx, cy, w, h, entry, class_idx, class_probs, objectness, threshold=0.5):
        self.cx = cx
        self.cy = cy
        self.in_w = in_w
        self.in_h = in_h
        self.w = w
        self.h = h
        self.entry = entry
        self.class_idx = class_idx
        self.class_probs = class_probs
        self.probs = class_probs[class_idx] * objectness

    def to_xyxy(self) -> tuple[tuple[int, int], tuple[int, int]]:
        hw, hh = self.w / 2, self.h / 2
        x1, x2 = int((self.cx - hw) * self.in_w), int((self.cx + hw) * self.in_w)
        y1, y2 = int((self.cy - hh) * self.in_h), int((self.cy + hh) * self.in_h)
        return (x1, y1), (x2, y2)
    
    def to_xywh(self) -> tuple[int, int, int, int]:
        hw, hh = self.w / 2, self.h / 2
        x, y = int((self.cx - hw) * self.in_w), int((self.cy - hh) * self.in_h)
        w, h = int(self.w * self.in_w), int(self.h * self.in_h)
        return x, y, w, h    

    def to_yolo(self) -> tuple[int, int, int, int]:
        cx, cy = int(self.cx * self.in_w), int(self.cy * self.in_h)
        w, h = int(self.w * self.in_w), int(self.h * self.in_h)
        return cx, cy, w, h

# YOLO
def entry_index(layer, batch: int, location: int, entry: int) -> int:
    n = location // (layer.out_w * layer.out_h)
    loc = location % (layer.out_w * layer.out_h)
    grid = layer.out_w * layer.out_h
    classes = layer.classes

    return (batch * grid * layer.out_c) + (n * grid * (5 + classes)) + (entry * grid) + loc

# Layer, Anchor, Index, position_x, position_y, network_input_width, network_input_height
def get_yolo_box(layer, n: int, entry: int, px: int, py: int, in_w: int, in_h: int, threshold: float) -> BBox:
    box = None
    out = layer.get_output()
    stride = layer.out_w * layer.out_h
    # Layers output
    x, y, w, h, o = [out[entry + i * stride] for i in range(5)]
    if layer.new_coords == 0:
        cx = (px + x) / layer.out_w
        cy = (py + y) / layer.out_h
        cw = math.exp(w) * layer.biases[2 * n] / in_w
        ch = math.exp(h) * layer.biases[2 * n + 1] / in_h
        probs = layer.get_class_probs(out, entry)
        class_idx = probs.index(max(probs))
        box = BBox(in_w, in_h, cx, cy, cw, ch, entry, class_idx, probs, o, threshold)
    else:
        pass
    return box

# Get BBox IoU
def bbox_iou(bbox1: BBox, bbox2: BBox) -> float:
    p1, p2 = bbox1.to_xyxy()
    box1 = p1 + p2
    p1, p2 = bbox2.to_xyxy()
    box2 = p1 + p2
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Compute the area of intersection rectangle
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    # Compute the Intersection over Union
    iou = intersection / float(box1_area + box2_area - intersection)
    return iou

# non_maximum_suppression
def non_maximum_suppression_bboxes(bboxes: list[BBox], iou_threshold: float=0.5) -> list[BBox]:
    assert len(bboxes) >= 2, "At least 2 BBox required."
    nms_bboxes = []

    sorted_bboxes = sorted(bboxes, key=lambda x: x.probs, reverse=True)
    
    while sorted_bboxes:
        best_bbox = sorted_bboxes.pop(0)
        nms_bboxes.append(best_bbox)
        # Remove
        remove_bboxes = [bbox for bbox in sorted_bboxes if bbox_iou(best_bbox, bbox) > iou_threshold]
        for bbox in remove_bboxes:
            sorted_bboxes.remove(bbox)
    
    return nms_bboxes