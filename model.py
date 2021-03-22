import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
DEVICE = 'cuda:3' if torch.cuda.is_available() else 'cpu'

def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.zeros_(m.weight)
            m.bias.data.fill_(0.0)

class Yolo(nn.Module):
    
    def __init__(self, grid_size, num_boxes, num_classes):
        super(Yolo, self).__init__()
        vgg16 = models.vgg16(True)
        features = list(vgg16.features.children())
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        self.features = nn.Sequential(
            *features
        )
        self.detector = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.S*self.S*(self.B*5+self.C))
        )
        self.detector.apply(init_weights)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.detector(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.S, self.S, self.B*5+self.C)
        return x




class Loss(nn.Module):
    def __init__(self, grid_size=7, num_bboxes=2, num_classes=20, l_coord=5, l_noobj=0.5):
        """ Loss module for Yolo v1.
        Use grid_size, num_bboxes, num_classes information if necessary.

        Args:
            grid_size: (int) size of input grid.
            num_bboxes: (int) number of bboxes per each cell.
            num_classes: (int) number of the object classes.
        """
        super(Loss, self).__init__()
        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, bbox1, bbox2):
        """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
        Use this function if necessary.

        Args:
            bbox1: (Tensor) bounding bboxes, sized [N, 4].
            bbox2: (Tensor) bounding bboxes, sized [M, 4].
        Returns:
            (Tensor) IoU, sized [N, M].
        """
        N = bbox1.size(0)
        M = bbox2.size(0)

        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Conpute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt   # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0 # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter # [N, M]
        iou = inter / union           # [N, M]

        return iou

    def forward(self, pred_tensor, target_tensor):
        """ Compute loss.

        Args:
            pred_tensor (Tensor): predictions, sized [batch_size, S, S, Bx5+C], 5=len([x, y, w, h, conf]).
            target_tensor (Tensor):  targets, sized [batch_size, S, S, Bx5+C].
        Returns:
            loss_xy (Tensor): localization loss for center positions (x, y) of bboxes.
            loss_wh (Tensor): localization loss for width, height of bboxes.
            loss_obj (Tensor): objectness loss.
            loss_noobj (Tensor): no-objectness loss.
            loss_class (Tensor): classification loss.
        """
        # Write your code here
        N = pred_tensor.size()[0]
        # contain obj
        coo_mask = target_tensor[:, :, :, 4] > 0 # 7x7 plane
        # no obj
        noo_mask = target_tensor[:, :, :, 4] == 0

        coo_mask = coo_mask.bool()
        noo_mask = noo_mask.bool()

        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor) #target과 같은 사이즈로 확장
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        # coo_pred：tensor[, 30]
        coo_pred = pred_tensor[coo_mask].view(-1, 30).to(DEVICE) # pred_tensor coo필터링 후 펼치기 ? x 30
        # box[x1,y1,w1,h1,c1], [x2,y2,w2,h2,c2] => ? x 5
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5).to(DEVICE)# box정보와 확률정보 분리
        # class
        class_pred = coo_pred[:, 10:] #?x20
        
        coo_target = target_tensor[coo_mask].view(-1, 30).to(DEVICE) # 동일하게 coo_mask 펼치기 ? x 30
        box_target = coo_target[:, :10].contiguous().view(-1, 5).to(DEVICE) # 동일하게 box정보와 확률정보 분리
        class_target = coo_target[:, 10:]
        

        ######### I. compute not contain obj loss############
        noo_pred = pred_tensor[noo_mask].view(-1, 30).to(DEVICE) # 각 tensor에서 no obj인 경우들의 값들만 남기고
        noo_target = target_tensor[noo_mask].view(-1, 30).to(DEVICE) # 형태도 펼침 ? x 30

        noo_pred_mask = torch.ByteTensor(noo_pred.size()).to(DEVICE)
        noo_pred_mask.zero_() # no obj의 confidence score만 비교
        noo_pred_mask[:, 4] = 1# 그를 위한 마스크를 만드는 과정
        noo_pred_mask[:, 9] = 1
        noo_pred_mask = noo_pred_mask.bool()

        noo_pred_c = noo_pred[noo_pred_mask].to(DEVICE) # confidence만 추출
        noo_target_c = noo_target[noo_pred_mask].to(DEVICE)
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, reduction='sum') #loss 계산

        ######### II. compute contain obj loss #############
        coo_response_mask = torch.ByteTensor(box_target.size()).to(DEVICE)
        coo_response_mask.zero_()
        coo_not_response_mask = torch.ByteTensor(box_target.size()).to(DEVICE)
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).to(DEVICE) #


        #### 0. choose the best iou box
        for i in range(0, box_target.size()[0], 2):  
            box1 = box_pred[i:i + 2]  #  한 그리드당 2개 박스 이므로 2개씩 할당
            box1_xyxy = torch.FloatTensor(box1.size())
            # (x,y,w,h) -> x,y,x,y로 변환 iou를 구하기 위해서
            box1_xyxy[:, :2] = box1[:, :2] / self.S - 0.5 * box1[:, 2:4] # remember delta_xy = (cxcy_sample -xy)/(1/S)
            box1_xyxy[:, 2:4] = box1[:, :2] / self.S + 0.5 * box1[:, 2:4] # 따라서 지금 값에 1/s를 곱해야 전체 이미지에서의 비율 나옴
            
            box2 = box_target[i:i + 2] #참고 target box는 둘다 같은 box정보
            box2_xyxy = torch.FloatTensor(box2.size())
            box2_xyxy[:, :2] = box2[:, :2] / self.S - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] / self.S + 0.5 * box2[:, 2:4]

            # iou(pred_box[2,], target_box[2,])
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])
            max_iou, max_index = iou.max(0) # idx는 0 or 1
            # print(f'max_iou:{max_iou}, max_index:{max_index}')
            max_index = max_index.to(DEVICE)

            coo_response_mask[i + max_index] = 1 #iou가 큰 해당 idx 박스값을 모두 1로 변환 ?x5
            coo_not_response_mask[i + 1 - max_index] = 1 # 반대의 경우를 모두 1로 변환
            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            box_target_iou[i + max_index, torch.LongTensor([4]).to(DEVICE)] = max_iou.to(DEVICE) 
            #confidence 위치에 최대 iou값 대입 (Pr(obj)는 1이므로 gt의 confidence값 완성)
        
        box_target_iou = box_target_iou.to(DEVICE)

        # 1.response loss (responsible 한 box에 대한 loss)
        box_pred_response = box_pred * coo_response_mask
        box_target_response_iou = box_target_iou * coo_response_mask # iou를 포함한 confidence score 정보만 있음
        box_target_response = box_target * coo_response_mask

        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], reduction='sum')
        
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], reduction='sum') + F.mse_loss(
            torch.sqrt(box_pred_response[:, 2:4] + 1e-6), torch.sqrt(box_target_response[:, 2:4] +  1e-6), reduction='sum')
        #sqrt시 수식적으로 말이 안되는 경우를 방지하기 위한 작은 수를 더해야함

        # 2.class loss
        class_loss = F.mse_loss(class_pred, class_target, reduction='sum')

        return (self.l_coord * loc_loss +  contain_loss + self.l_noobj * nooobj_loss + class_loss) / N