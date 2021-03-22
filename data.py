import os
import cv2
import random
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class VOCDetection(Dataset):
    def __init__(self, root, split='train', image_size=224):
        assert image_size == 224, 'Currently, only image of 448 is supported'

        self.root = root
        self.split = split
        self.image_size = image_size
        self.fnames, self.boxes, self.labels = self.parse_labels()

    def parse_labels(self):
        label_file_path = os.path.join(self.root, 'labels/%s.txt' % (self.split))
        with open(label_file_path) as f:
            fnames, boxes, labels = [], [], []
            for line in f.readlines():
                splited = line.strip().split()
                fnames.append(splited[0])
                num_boxes = (len(splited)-1)//5
                box = []
                label = []
                for i in range(num_boxes):
                    x = float(splited[1+5*i])
                    y = float(splited[2+5*i])
                    x2 = float(splited[3+5*i])
                    y2 = float(splited[4+5*i])
                    c = splited[5+5*i]
                    box.append([x, y, x2, y2])
                    label.append(int(c)+1)#30개 벡터중에서
                    #클래스확률 부분에 해당 클래스idx를 1로 설정하기 위해 1을 더함(위치상)
                boxes.append(torch.Tensor(box))
                labels.append(torch.LongTensor(label))
        return fnames, boxes, labels

    def transform(self, img):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ])
        return transform(img)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root, 'images', fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        if self.split == 'train':
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.randomScale(img,boxes)
            img = self.randomBlur(img)
            img = self.RandomBrightness(img)
            img = self.RandomHue(img)
            img = self.RandomSaturation(img)
            img,boxes,labels = self.randomShift(img,boxes,labels)
            img,boxes,labels = self.randomCrop(img,boxes,labels)

        h,w,_ = img.shape #채널은 불필요
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)# 독립적 coord를 전체이미지 사이즈에 대한 비율로 변환
        img = self.BGR2RGB(img) 
        img = cv2.resize(img,(self.image_size,self.image_size))
        target = self.encoder(boxes,labels)
        img = self.transform(img)
        return img,target

    def __len__(self):
        return len(self.boxes)

    def encoder(self,boxes,labels):
        grid_num = int(self.image_size / 32) 
        target = torch.zeros((grid_num,grid_num,30))# output tensor와 사이즈 동일하게 하기 위해 0초기화된 확장 텐서 생성
        cell_size = 1./grid_num
        wh = boxes[:,2:]-boxes[:,:2]
        cxcy = (boxes[:,2:]+boxes[:,:2])/2 #Nx2 각 박스의 센터들
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample/cell_size).ceil()-1 #그리드 인덱스. ceil은 올림
            target[int(ij[1]),int(ij[0]),4] = 1 # 각 그리드에 있는 30개 텐서를 손질
            target[int(ij[1]),int(ij[0]),9] = 1 # 컨피던스는 1이 이상적
            target[int(ij[1]),int(ij[0]),int(labels[i])+9] = 1 # gt 클래스 1로 설정
            xy = ij*cell_size # 각자의 그리드의 왼쪽위 모서리값 출력(전체 이미지 비율 기준)
            delta_xy = (cxcy_sample -xy)/cell_size
            target[int(ij[1]),int(ij[0]),2:4] = wh[i] # 전체이미지 기준 비율의 wh값들입력
            target[int(ij[1]),int(ij[0]),:2] = delta_xy # 그리드 내의 offset값
            target[int(ij[1]),int(ij[0]),7:9] = wh[i]
            target[int(ij[1]),int(ij[0]),5:7] = delta_xy
        return target

    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    
    def RandomBrightness(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self,bgr):
        if random.random()<0.5:
            bgr = cv2.blur(bgr,(5,5))
        return bgr

    def randomShift(self,bgr,boxes,labels):
        center = (boxes[:,2:]+boxes[:,:2])/2
        if random.random() <0.5:
            height,width,c = bgr.shape
            after_shfit_image = np.zeros((height,width,c),dtype=bgr.dtype)
            after_shfit_image[:,:,:] = (104,117,123) #이동후 빈 공간 색이 이것
            shift_x = random.uniform(-width*0.2,width*0.2)
            shift_y = random.uniform(-height*0.2,height*0.2)
            if shift_x>=0 and shift_y>=0:
                after_shfit_image[int(shift_y):,int(shift_x):,:] = bgr[:height-int(shift_y),:width-int(shift_x),:]
            elif shift_x>=0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width-int(shift_x),:]
            elif shift_x <0 and shift_y >=0:
                after_shfit_image[int(shift_y):,:width+int(shift_x),:] = bgr[:height-int(shift_y),-int(shift_x):,:]
            elif shift_x<0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = bgr[-int(shift_y):,-int(shift_x):,:]

            shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:,0] >0) & (center[:,0] < width) #센터가 이미지 바깥으로 나갔는지 검사
            mask2 = (center[:,1] >0) & (center[:,1] < height)
            mask = (mask1 & mask2).view(-1,1) # 둘다 만족해야함
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4) # True False를 idx위치에 넣으면 False위치 원소는 사라지고 True 위치의 원소만 나옴
            if len(boxes_in) == 0: # 센터가 넘어가므로 변환 불가
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in+box_shift
            labels_in = labels[mask.view(-1)] # 센터가 안넘어간 레이블만 반환
            return after_shfit_image,boxes_in,labels_in
        return bgr,boxes,labels

    def randomScale(self,bgr,boxes):
        if random.random() < 0.5:
            scale = random.uniform(0.8,1.2)
            height,width,c = bgr.shape
            bgr = cv2.resize(bgr,(int(width*scale),height))
            scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr,boxes
        return bgr,boxes

    def randomCrop(self,bgr,boxes,labels):
        if random.random() < 0.5:
            center = (boxes[:,2:]+boxes[:,:2])/2
            height,width,c = bgr.shape
            h = random.uniform(0.6*height,height)
            w = random.uniform(0.6*width,width)
            x = random.uniform(0,width-w)
            y = random.uniform(0,height-h)
            x,y,h,w = int(x),int(y),int(h),int(w)

            center = center - torch.FloatTensor([[x,y]]).expand_as(center)
            mask1 = (center[:,0]>0) & (center[:,0]<w)
            mask2 = (center[:,1]>0) & (center[:,1]<h)
            mask = (mask1 & mask2).view(-1,1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if(len(boxes_in)==0):
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:,0]=boxes_in[:,0].clamp_(min=0,max=w)
            boxes_in[:,2]=boxes_in[:,2].clamp_(min=0,max=w)
            boxes_in[:,1]=boxes_in[:,1].clamp_(min=0,max=h)
            boxes_in[:,3]=boxes_in[:,3].clamp_(min=0,max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y+h,x:x+w,:]
            return img_croped,boxes_in,labels_in

        return bgr,boxes,labels

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta,delta)
            im = im.clip(min=0,max=255).astype(np.uint8)
        return im


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from tqdm import tqdm # for 문의 상태바를 보여줌
    root = 'dataset'

    train_dset = VOCDetection(root, 'train')
    train_loader = DataLoader(train_dset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
    for item in tqdm(train_loader):
        pass

    test_dset = VOCDetection(root, 'test')
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
    for item in tqdm(test_loader):
        pass
