import os
from tkinter import Image
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import PIL
from PIL import Image
import openpifpaf
from torchvision.transforms.functional import to_pil_image
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import json
from sklearn.utils import resample

_target = np.array(['front', 'back', 'left', 'right'])

class TestDataset(Dataset):
    def __init__(self, img_dir, img_size=(150, 80)):
        self.target = _target
        imgs = []
        labels = []
        for l in self.target:
            for img in os.listdir(os.path.join(img_dir, l)):
                imgs += [os.path.join(img_dir, l, img)]
                labels += [l]

        self.imgs = imgs
        self.labels = labels
        self.resize = transforms.Resize(img_size)

        self.predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image = read_image(self.imgs[index])
        image = self.resize(image)

        label = self.labels[index]
        label = self.target == label
        label = torch.tensor(label).float()
        
        pil_im =to_pil_image(image).convert('RGB')
        
        predictions, _, _ = self.predictor.pil_image(pil_im)
        
        if not predictions:
            return -1

        skt = np.append(predictions[0].data, (predictions[0].data[5] + predictions[0].data[6])/2).reshape(-1, 3)
        # skt[:, 1:2] = skt[:, 1:2]/150
        # skt[:, 0:1] = skt[:, 0:1]/80

        return skt, label

    

class TestRawData(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.images = os.listdir(dir)
        self.predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_dir = os.path.join(self.dir, self.images[index])
        
        image = Image.open(image_dir)
        
        predictions, _, _ = self.predictor.pil_image(image)
        width, height = image.size
        if not predictions:
            return -1

        skt = np.append(predictions[0].data, (predictions[0].data[5] + predictions[0].data[6])/2).reshape(-1, 3)
        

        return skt, image_dir

class TrainRawData(Dataset):
    def __init__(self, dir, img_size = (150, 80)):
        self.target = _target
        self.dir = dir
        self.img_size = img_size
        imgs = []
        labels = []
        for t in self.target:
            image_files = os.listdir(os.path.join(dir, t))
            for file in image_files:
                imgs += [os.path.join(dir, t, file)]
                labels += [t]

        self.images = imgs
        self.labels = labels
        self.resize = transforms.Resize(self.img_size)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = read_image(self.images[index])        
        img = self.resize(img)
        label = self.labels[index]

        return img, label

def mk_skeleton(dataset, dir = './'):
    predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')
    skeletons = []
    labels = []

    target = np.array(['front', 'back', 'left', 'right'])
    for k, (i, l) in enumerate(dataset):
        image = to_pil_image(i)
        predictions, _, _ = predictor.pil_image(image)
        label = target == l
        if not predictions:
            continue

        skt = np.append(predictions[0].data, (predictions[0].data[5] + predictions[0].data[6])/2).reshape(-1, 3)
        # skt[:, 1:2] = skt[:, 1:2]/150
        # skt[:, 0:1] = skt[:, 0:1]/80


        if len(skeletons) ==0:
            skeletons = skt[np.newaxis, :]
            labels = label[np.newaxis, :]
        else :
            skeletons = np.append(skeletons, skt[np.newaxis, :], axis = 0)
            labels = np.append(labels, label[np.newaxis, :], axis = 0)

    print('save file')
    np.save(os.path.join(dir, "./skeletons"),skeletons)
    print('skeletons.npy done')
    np.save(os.path.join(dir, './labels'), labels)
    print('labels.npy done')


class TrainSkeletonData(Dataset):
    def __init__(self, dir, make_skeleton = False):
        if make_skeleton:
            print("make_skeleton")
            dataset =TrainRawData(dir)
            mk_skeleton(dataset)

        self.target = _target
        self.skeletons = np.load(os.path.join("./", "skeletons.npy"))
        self.labels = np.load(os.path.join("./", 'labels.npy'))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        skeleton = self.skeletons[index]
        label = self.labels[index].astype(float)

        return skeleton, label


class AlphaposeSkeletonData(Dataset):
    def __init__(self, json_dir, image_dir):
        skeletons = []
        labels = []
        for pose_name in os.listdir(json_dir):
            json_file = os.path.join(json_dir, pose_name)
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            dic = {}
            for data in json_data:
                cls = data['cls']
                if cls != 0:
                    continue
                image_id = data['image_id']
                box = data['box']
                keypoints = np.array(data['keypoints']).reshape(-1, 3)
                x1, y1, x2, y2 = box
                temp = (x2 - x1) * (y2 - y1)
                if  temp < 0:
                    temp *= -1
                
                keypoints = np.append(keypoints,(keypoints[5:6] + keypoints[6:7])/2, axis = 0)
                if image_id in dic.keys():
                    if temp > dic[image_id]['box']:
                        dic[image_id]['box'] = temp
                        dic[image_id]['keypoints'] = keypoints
                                     
                else :                     
                    dic[image_id] = {"keypoints" : keypoints, "box" : temp, 'label' : (_target == pose_name[:-5]).astype('int')}
            for id, values in dic.items():
                skeletons += [values['keypoints']]
                labels += [values['label']]
        skeletons = np.array(skeletons)
        labels = np.array(labels)
        
        min_len = min((labels[:,0]==1).sum(),
        (labels[:,1]==1).sum(),
        (labels[:,2]==1).sum(),
        (labels[:,3]==1).sum())

        _skeletons =[]
        _labels = []
        for ii in range(0, 4):
            sk_idx = [np.random.choice(range(len(skeletons[labels[:, ii]==1])), size = min_len, replace = False)]
            _skeletons+=[skeletons[labels[:, ii]==1][sk_idx[0]]]
            _labels+=[labels[labels[:, ii]==1][sk_idx[0]]]
        skeletons = np.array(_skeletons).reshape(-1,18, 3)
        labels = np.array(_labels).reshape(-1, 4)
        self.skeletons = torch.tensor(skeletons).float()
        self.labels = torch.tensor(labels).float()
        


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        skeleton = self.skeletons[index]
        label = self.labels[index]

        return skeleton, label

class TestAlphaPose(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        dic = {}
        for data in json_data:
            cls = data['cls']
            if cls != 0:
                continue
            image_id = data['image_id']
            box = data['box']
            keypoints = np.array(data['keypoints']).reshape(-1, 3)
            x1, y1, x2, y2 = box
            temp = (x2 - x1) * (y2 - y1)
            if  temp < 0:
                temp *= -1
            
            keypoints = np.append(keypoints,(keypoints[5:6] + keypoints[6:7])/2, axis = 0)
            if image_id in dic.keys():
                if temp > dic[image_id]['box']:
                    dic[image_id]['box'] = temp
                    dic[image_id]['keypoints'] = keypoints
                                    
            else :                     
                dic[image_id] = {"keypoints" : keypoints, "box" : temp}
        self.dic = dic


    def __len__(self):
        return len(self.dic.keys())

    def __getitem__(self, index):
        image_id = list(self.dic.keys())[index]
        skeleton = self.dic[image_id]['keypoints']

        return skeleton, image_id