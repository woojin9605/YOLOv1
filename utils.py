import os
import cv2
import numpy as np

import torch
from torchvision import transforms



DEVICE = 'cuda:3' if torch.cuda.is_available() else 'cpu'


VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']

def NMS(bboxes, scores, threshold=0.35):
    ''' Non Max Suppression
    Args:
        bboxes: (torch.tensors) list of bounding boxes. size:(N, 4) ((left_top_x, left_top_y, right_bottom_x, right_bottom_y), (...))
        probs: (torch.tensors) list of confidence probability. size:(N,) 
        threshold: (float)   
    Returns:
        keep_dim: (torch.tensors)
    '''
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True) #confidence 스코어 내림차순 정렬 idx얻음
    order = order.squeeze()
    keep = []
    while order.numel() > 0: #조건 만족 까지 무한반복예정 numel은 원소개수
        try:
            i = order[0]
        except:
            i = order.item()
        keep.append(i)

        if order.numel() == 1: break #단 한개의 idx만 남을때까지 무한반복예정
        #confi 가장 큰 박스와 나머지 박스들의 iou 비교 예정
        xx1 = torch.minimum(x1[order[1:]], x1[i])
        yy1 = torch.minimum(y1[order[1:]], y1[i])
        xx2 = torch.maximum(x2[order[1:]], x2[i])
        yy2 = torch.maximum(y2[order[1:]], y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ovr = ovr.squeeze()
        ids = torch.nonzero((ovr <= threshold)).squeeze() # 스레숄드 만족 못하는 박스들의 idx
        if ids.numel() == 0: # 스레숄드 만족 못하는 박스들을 가져가야함
            break
        order = order[ids + 1] # 맨처음 원소는 비교대상이므로 1 더해서 제외
    keep_dim = torch.LongTensor(keep) #최종적으로 가장 큰 confi가지던 원소부터 쭉
    return keep_dim #비교해 나가서 해당 idx들만 되돌림


def decoder(pred):
    """ Decoder function that decode the output-pred to bounding box, class and probability. 
    Args:
        pred: (torch.tensors) 1x7x7x30
    Returns:
        bboxes: (torch.tensors) list of bounding boxes. size:(N, 4) ((left_top_x, left_top_y, right_bottom_x, right_bottom_y), (...))
        class_idxs: (torch.tensors) list of class index. size:(N,)
        probs: (torch.tensors) list of confidence probability. size:(N,)
    """
    grid_num = 7
    bboxes = []
    class_idxs = []
    probs = []
    cell_size = 1./grid_num
    pred = pred.data
    pred = pred.squeeze(0) #7x7x30
    contain1 = pred[:,:,4].unsqueeze(2) #confi값들만 7x7x1
    contain2 = pred[:,:,9].unsqueeze(2)
    contain = torch.cat((contain1,contain2),2) # 7x7x2
    mask1 = contain > 0.1
    mask2 = (contain==contain.max()) #we always select the best contain_prob what ever it>0.9
    mask = (mask1+mask2).gt(0) #0.1이상 이거나 둘중에 큰 confi값 찾는 마스크. 0이상인 값들만 True 아니면 False
    min_score,min_index = torch.min(contain,2)

    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                index = min_index[i,j]
                mask[i,j,index] = 0
                if mask[i,j,b] == 1:
                    box = pred[i,j,b*5:b*5+4] #coord
                    contain_prob = torch.FloatTensor([pred[i,j,b*5+4]]) #confi score
                    xy = torch.FloatTensor([j,i])*cell_size #up left of cell
                    box[:2] = box[:2]*cell_size + xy # return cxcy relative to image
                    box_xy = torch.FloatTensor(box.size())#convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    box_xy[2:] = box[:2] + 0.5*box[2:]
                    max_prob,cls_index = torch.max(pred[i,j,10:],0)
                    if float((contain_prob*max_prob)[0]) > 0.1: #confi x class_prob > 0.1
                        bboxes.append(box_xy.view(1,4))
                        class_idxs.append(cls_index)
                        probs.append(contain_prob*max_prob)


    if len(bboxes) == 0: # Any box was not detected
        bboxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        class_idxs = torch.zeros(1)
     
    else: 
        #list of tensors -> tensors
        bboxes = torch.stack(bboxes).squeeze(1)
        probs = torch.stack(probs).squeeze(0)
        class_idxs = torch.stack(class_idxs)    
    keep_dim = NMS(bboxes, probs, threshold=0.35) # Non Max Suppression
    return bboxes[keep_dim].squeeze(1), class_idxs[keep_dim], probs[keep_dim]


def inference(model, image_path):
    """ Inference function
    Args:
        model: (nn.Module) Trained YOLO model.
        image_path: (str) Path for loading the image.
    """
    # load & pre-processing
    
    image_name = image_path.split('/')[-1]
    image = cv2.imread(image_path)

    h, w, c = image.shape
    img = cv2.resize(image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = transform(torch.from_numpy(img).float().div(255).transpose(2, 1).transpose(1, 0)) #Normalization
    img = img.unsqueeze(0)
    img = img.to(DEVICE)

    # inference
    output_grid = model(img).cpu()
    # decode the output grid to the detected bounding boxes, classes and probabilities.
    bboxes, class_idxs, probs = decoder(output_grid)
    num_bboxes = bboxes.size(0)

    # draw bounding boxes & class name
    for i in range(num_bboxes):
        bbox = bboxes[i]
        class_name = VOC_CLASSES[int(class_idxs[i])]
        prob = probs[i]

        x1, y1 = int(bbox[0] * w), int(bbox[1] * h)
        x2, y2 = int(bbox[2] * w), int(bbox[3] * h)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, '%s: %.2f'%(class_name, prob), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, 8)


    cv2.imwrite(image_name.replace('.jpg', '_result.jpg'), image)