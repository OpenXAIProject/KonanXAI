import time
import torch
import torchvision
from pathlib import Path
import os,sys
def non_max_suppression(prediction, logits, conf_thres=0.45, iou_thres=0.45, classes=None, agnostic=False,
                            multi_label=False, labels=(), max_det=300, model_name: str = None):
    """Runs Non-Maximum Suppression (NMS) on inference and logits results

    Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls] and pruned input logits (n, number-classes)
    """
    yolo_dict = {
        'Yolov5s': '~/.cache/torch/hub/ultralytics_yolov5_master',
    }
    yolo_model = Path(os.path.expanduser(yolo_dict[model_name]))
    sys.path.append(str(yolo_model))
    from utils.metrics import box_iou
    from utils.general import xywh2xyxy
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
    box_index = []
    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    logits_output = [torch.zeros((0, 80), device=logits.device)] * logits.shape[0]
    for xi, (x, log_) in enumerate(zip(prediction, logits)):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        
        ##detect box
        bbox_index = ((xc[0]==True).nonzero(as_tuple=True)[0])
        ##
        x = x[xc[xi]]  # confidence
        log_ = log_[xc[xi]]
        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        # log_ *= x[:, 4:5]
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            #####
            remove_index = (([conf.view(-1) > conf_thres][0]==False).nonzero(as_tuple=True)[0]).tolist()
            mask = torch.ones_like(bbox_index, dtype=torch.bool)
            mask[remove_index] = False
            filter_bbox_index = bbox_index[mask]
            #####
            # log_ = x[:, 5:]
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            log_ = log_[conf.view(-1) > conf_thres]
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        #####
        # mask = torch.zeros_like(filter_bbox_index, dtype=torch.bool)
        # for b_idx in i:
        #     box_index.append(filter_bbox_index[b_idx.item()].item())
        [box_index.append(filter_bbox_index[b_idx.item()].item()) for b_idx in i]
        
        #####
        
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        logits_output[xi] = log_[i]
        assert log_[i].shape[0] == x[i].shape[0]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output, logits_output, box_index

def yolo_choice_layer(dbox, select_layers):
        index_len = []
        index_tmep =[]
        for i in dbox:
            i = i.shape
            index_len.append(i[1]*i[2]*i[3])
        for select_layer in select_layers:
            if index_len[0]>select_layer:
                index_tmep.append(0)
            elif index_len[0]+index_len[1]> select_layer:
                index_tmep.append(1)
            else:
                index_tmep.append(2)
        return index_tmep