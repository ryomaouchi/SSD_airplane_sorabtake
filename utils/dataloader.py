import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
import xml.etree.ElementTree as ET
from utils.augmentations import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

    
class xml_to_list(object):
    # 1枚の画像に対するxml形式のアノテーションデータを、画像サイズで規格化してからリスト形式に変換。
    
    def __init__(self, classes):
        self.classes = classes
    
    def __call__(self, xml_path, width, height):
        
        ret = []
        xml = ET.parse(xml_path).getroot()
        
        for obj in xml.iter('object'):
            
            xmlbox = obj.find('bndbox2D')
            bb = [float(xmlbox.find('xmin').text),
                float(xmlbox.find('ymin').text),
                float(xmlbox.find('xmax').text),
                float(xmlbox.find('ymax').text)]
            
            bb[0] /= width
            bb[2] /= width
            
            bb[1] /= height
            bb[3] /= height
            
            label_idx = self.classes.index("airplane")
            bb.append(label_idx) # 0 は飛行機のラベル
            
            ret.append(bb)
         
        return np.array(ret)
    
    
class DataTransform():
    
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(),
                ToAbsoluteCoords(), 
                PhotometricDistort(), 
                Expand(color_mean), 
                RandomSampleCrop(), 
                RandomMirror(), 
                ToPercentCoords(), 
                Resize(input_size), 
                SubtractMeans(color_mean)
            ]),
            'val': Compose([
                ConvertFromInts(),
                Resize(input_size), 
                SubtractMeans(color_mean)
            ])
        }
        
    def __call__(self, img, phase, boxes, labels):
        """"
        phase = 'train' or 'val'
        """
        
        return self.data_transform[phase](img, boxes, labels)
    

    
class RareplanesDataset(data.Dataset):
    
    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.transform_anno = transform_anno
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt
        
    def pull_item(self, index):
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)
        height, width, channels = img.shape
        
        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height)
        
        img, boxes, labels = self.transform(
            img, self.phase, anno_list[:,:4], anno_list[:,4]
        )
        
        img = torch.from_numpy(img[:,:,(2,1,0)]).permute(2,0,1)
        
        gt = np.hstack((boxes, np.expand_dims(labels, axis = 1)))
        
        return img, gt, height, width

    
def get_color_mean(train_img_paths):   
    
    channel_mean = np.zeros((3,len(train_img_paths)))
    out_mean = np.zeros(3)
    
    for i in range(len(train_img_paths)):
        img = cv2.imread(train_img_paths[i])
        channel_mean[:,i] = np.mean(img, axis=(0,1))
            
    out_mean = np.mean(channel_mean, axis= 1)
    return (out_mean[0], out_mean[1], out_mean[2])
    
    
    
def od_collate_fn(batch):
    
    targets = []
    imgs = []
    
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        
    imgs = torch.stack(imgs, dim=0)
    return imgs, targets
