import KonanXAI as XAI
from KonanXAI.lib.core import darknet
import cv2
import numpy as np
def get_yolo_box_test(x, bias, n, index, i, j, lw, lh, w, h, stride, new_coords):
    box = {}
    if new_coords != 0:
        pass
    else:
        box['x'] = (i + x[index + 0 * stride]) / lw
        box['y'] = (j + x[index + 1 * stride]) / lh
        box['w'] = math.exp(x[index + 2 * stride]) * bias[2 * n] / w
        box['h'] = math.exp(x[index + 3 * stride]) * bias[2 * n + 1] / h
    return box['x'], box['y'], box['w'], box['h']

def entry_index(l, batch, location, entry, classes):
    n = location // (l['w']*l['h'])
    loc = location % (l['w']*l['h'])
    return int(batch*(l['w']*l['h']*l['c']) + n * l['w'] * l['h'] * (4+classes+1) + entry*l['w']*l['h'] + loc)

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
img = darknet.open_image(r"D:\xai_refactoring\test.jpg", (416, 416))
net: darknet.Network = xai.model.net
results = net.predict_using_gradient_hook(img)
print(results)
# yolo2 = results['layers_output']['layer_30']['data']
# yolo2 = results['layers_output']['layer_37']['data']
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
    cx, cy, bw, bh = bbox[:]
    x1, y1 = cx - bw/2, cy - bh/2
    x2, y2 = cx + bw/2, cy + bh/2

    x1, y1, x2, y2 = list(map(lambda x: round(x), (x1*width, y1*height, x2*width, y2*height)))
    return x1, y1, x2, y2

img = cv2.imread(r"D:\xai_refactoring\test.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (416, 416))
thres = 0.2

coord = 4
prob = 1
netw = 416
neth = 416
box_wh = [13,26]
# width = 26
# height = 26
# width = 13
# height = 13
# size2d = width*height
size2d = [box_wh[0]**2,box_wh[1]**2]
classes = 80
features = coord + prob + classes
anchor_offset= [3,1]
# anchor_offset = 3
# anchor_offset = 1
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


# l = results['layers_output']['layer_37']
l_30 = results['layers_output']['layer_30']
l_37 = results['layers_output']['layer_37']
layers = [l_30, l_37]
max_box = None
box2, n = net.get_network_boxes(416, 416, thresh=thres)
for _ in range(n):
    dbbox = box2[_].bbox
    dbbox.x = dbbox.x
    dbbox.y = dbbox.y
    whalf = dbbox.w / 2
    hhalf = dbbox.h / 2
    cv2.rectangle(img, pt1=(int(dbbox.x - whalf), int(dbbox.y - hhalf)), pt2=(int(dbbox.x + whalf), int(dbbox.y + hhalf)), color=(255, 0, 0), thickness=2)
box_li = []
for k, l in enumerate(layers):
    for i in range(size2d[k]):
        row = i // l['w']
        col = i % l['w']
        for n in range(3):
            obj_index = entry_index(l, 0, n * l['w'] * l['h'] + i, 4, classes)
            objectness = l['data'][obj_index]
            if (objectness > thres):
                box_index = entry_index(l, 0, n*l['w']*l['h'] + i, 0, classes)
                bbox = get_yolo_box_test(l['data'], l['bias'], anchor_offset[k] + n, box_index, col, row, l['w'], l['h'], netw, neth, size2d[k], 0)
                x1, y1, x2, y2 = yolo2xyxy(416, 416, bbox[:4])
                if k ==0:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0,0,255),thickness=1)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0,255,0),thickness=1)
                    
                probs = np.zeros(classes)
                for j in range(classes):
                    class_index = entry_index(l, 0, n*l['w']*l['h'] + i, 4 + 1 + j, classes)
                    prob = objectness * l['data'][class_index]
                    if prob > thres:
                        probs[j] = prob
                        bbox += (objectness,j, probs)
                box_li.append(bbox)

        
"""    
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
"""
resized = cv2.resize(img, (640, 480))
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.imshow('Image', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

pass
# explolation = xai.explain(etype, save_combine=True, save_heatmap=True)