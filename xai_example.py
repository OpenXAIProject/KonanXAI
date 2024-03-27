import KonanXAI as XAI
from KonanXAI.lib.core import darknet
import cv2

# model 
mtype = XAI.ModelType.Yolov4Tiny
# platform
platform = XAI.PlatformType.Darknet
# dataset
dtype = XAI.DatasetType.COCO
# explain
etype = XAI.ExplainType.GradCAM

xai = XAI.XAI()
xai.load_model_support(mtype, platform, pretrained=True)
# xai.load_dataset_support(dtype, maxlen=10, shuffle=False)
print(xai.model)
img = darknet.open_image("D:/Gitlab/xai_re/test.jpg", (416, 416))
net: darknet.Network = xai.model.net
results = net.predict_using_gradient_hook(img)
print(results)
yolo2 = results['layers_output']['layer_37']['data']

# def yolo2xywh(width, height, annotation):
#     # yolo: center_x, center_y, object_w, object_h  ex) 0.123 0.3123, 0.412, 0.787
#     # xywh: min_x, min_y, object_w, object_h        ex) 482, 374, 800, 340
#     # objects = ["S60", "M1939", "ZPU"]
#     # obj = objects[int(annotation[0])]

#     cx, cy, ow, oh = annotation[1:]
#     min_x, min_y = round((cx - (ow/2)) * width), round((cy - (oh/2)) * height)
#     ow, oh = round(ow * width), round(oh * height)
#     return min_x, min_y, ow, oh
def yolo2xyxy(width, height, bbox):
    """
    YOLO format use relative coordinates for annotation
    x = center_x, y = center_y
    """
    cx, cy, bw, bh = bbox
    x1, y1 = cx - bw/2, cy - bh/2
    x2, y2 = cx + bw/2, cy + bh/2

    x1, y1, x2, y2 = list(map(lambda x: round(x), (x1*width, y1*height, x2*width, y2*height)))
    return x1, y1, x2, y2

img = cv2.imread("D:/Gitlab/xai_re/test.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (416, 416))
thres = 0.9

coord = 4
prob = 1
width = 26
height = 26
# width = 13
# height = 13
size2d = width*height
classes = 80
features = coord + prob + classes
# anchor_offset = 3
anchor_offset = 1
anchor_box = [
    (10,14),
    (23,27),
    (37,58),
    (81,82),
    (135,169),
    (344,319)
]

import math
def get_yolo_box(feature, l, i, n):
    box = {}
    fx = i % l['w']
    fy = i // l['h']
    if l['new_coords'] != 0:
        pass
    else:
        box['x'] = (fx + feature[0]) / l['w']
        box['y'] = (fy + feature[1]) / l['h']
        box['w'] = math.exp(feature[2]) * l['bias'][2 * n] / 416
        box['h'] = math.exp(feature[3]) * l['bias'][2 * n + 1] / 416
    return box['x'], box['y'], box['w'], box['h']


l = results['layers_output']['layer_37']
max_box = None
box2, n = net.get_network_boxes(416, 416, thresh=0.9)
for _ in range(n):
    dbbox = box2[_].bbox
    dbbox.x = dbbox.x - 30
    dbbox.y = dbbox.y - 50
    cv2.rectangle(img, pt1=(int(dbbox.x), int(dbbox.y)), pt2=(int(dbbox.x + dbbox.w), int(dbbox.y + dbbox.h)), color=(255, 0, 0), thickness=2)
for n in range(3):
    abox = anchor_box[n + anchor_offset]
    for i in range(size2d):
        # location = n * size2d + i
        # anchor = location // size2d
        # loc = location % size2d
        # entry = 0
        yolo = []
        for f in range(features):
            idx = n * size2d * features + f * size2d + i
            yolo.append(yolo2[idx])
        # f = yolo[idx:idx+features]
        box = get_yolo_box(yolo, l, i, n+anchor_offset)
        # bbox = yolo[:4]
        # ofs_x, ofs_y = i % width, i // height
        # ofs_w, ofs_h = abox
        # bbox[0] = (ofs_x + bbox[0]) / width
        # bbox[1] = (ofs_y + bbox[1]) / height
        # bbox[2] = (bbox[2] ** 2) * 4 * ofs_w / 416
        # bbox[3] = (bbox[3] ** 2) * 4 * ofs_h / 416
        score = yolo[5]
        class_prob = yolo[6:]
        class_idx = class_prob.index(max(class_prob))
        if score > thres:
            x1, y1, x2, y2 = yolo2xyxy(416, 416, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), 2)
            print(box, score, class_idx)
        # if max_box is None:
        #     max_box = (box, score)
        # if score > max_box[1]:
        #     max_box = (box, score)

        # print(idx, f)

resized = cv2.resize(img, (640, 480))
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.imshow('Image', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

pass
# explolation = xai.explain(etype, save_combine=True, save_heatmap=True)