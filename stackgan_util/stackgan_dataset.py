import glob

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset
import csv
import math

######################################################data augmentation 加在 transform中

class FaceDataset(Dataset):
    def __init__(self, fnames, tags, transform, transform64, transform256):
        self.transform = transform
        self.transform64 = transform64
        self.transform256 = transform256
        self.fnames = fnames
        self.tags = tags
        self.num_samples = len(self.fnames)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        # 1. Load the image
        img = torchvision.io.read_image(fname)
        # 2. Resize and normalize the images using torchvision.
        img = self.transform(img)
        img64 = self.transform64(img)
        img256 = self.transform256(img)
        
        tag = self.tags[idx]
        
        hair_tag = tag['hair']
        eyes_tag = tag['eyes']
        eyepatch_tag = tag['eyepatch']
        
        return img64, img256, hair_tag, eyes_tag, eyepatch_tag

    def __len__(self):
        return self.num_samples

class rotate_crop(object):
    def __init__(self, degree, originsize):
        self.degree = degree
        self.originsize = originsize
    def __call__(self, sample):
        #input: img, output: img
        random_degree = transforms.RandomRotation.get_params([-self.degree, self.degree])
        
        
        img = transforms.v2.functional.rotate(sample, random_degree,  interpolation=transforms.InterpolationMode.BILINEAR, expand=True)
        
        rad = math.radians( abs(random_degree)+45 )    # sinx+cosx = sqrt(2) * sin( x + pi/4 ) = originlength
        crop_len= math.floor( self.originsize/ math.sin(rad) /math.sqrt(2) )
        img = transforms.v2.functional.center_crop(img, output_size=int(crop_len)) 
        img = transforms.v2.functional.resize(img, size=(self.originsize, self.originsize))
        
        return img

def get_dataset(root, tag_doc, hair_list, eyes_list, mode='1'):
    
    fnames= []
    taglist= []
    with open(tag_doc, newline='') as csvfile:
        rows = csv.reader(csvfile)
        rows =list(rows)
        
    def random_prob( length, mode):    
        if mode == '1':
            prob = torch.rand(length)
            prob = prob/torch.sum(prob)
            return prob
        
        elif mode == '2':
            prob = torch.rand(length-1)
            prob = torch.sort(prob).values
            return torch.cat((prob, torch.tensor([1]))) - torch.cat((torch.tensor([0]),prob))
    
        
    for row in rows:
        pic_id, tags = row
        
        tag_dict={}
        hair_tag= torch.zeros(len(hair_list))
        eyes_tag= torch.zeros(len(eyes_list))
        eyepatch_tag= torch.zeros(1)
        
        for i, color in enumerate(hair_list):
            if color in tags:
                hair_tag[i]=1.0
        
        for i, color in enumerate(eyes_list):
            if color in tags and 'bicolored eyes' not in tags:
                eyes_tag[i]=1.0
        
        if not ( sum(hair_tag)>1 or sum(eyes_tag)>1 ) and not ( sum(hair_tag)==0 and sum(eyes_tag)==0 ):
#            if sum(hair_tag) != 1:
#                hair_tag = random_prob(len(hair_list), mode)
#            if sum(eyes_tag) != 1:
#                eyes_tag = random_prob(len(eyes_list), mode)
            if 'eyepatch' in tags:
                eyepatch_tag = torch.ones(1)
            
            tag_dict['hair']     = hair_tag
            tag_dict['eyes']     = eyes_tag
            tag_dict['eyepatch'] = eyepatch_tag
            taglist.append(tag_dict)
            
            fnames.append(os.path.join(root, pic_id+'.jpg'))
                

    # 1. Resize the image to (64, 64)
    # 2. Linearly map [0, 1] to [-1, 1]
    
    compose = [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        rotate_crop(degree=15, originsize=256),
    ]
    
    compose64 = [
        #transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        #transforms.RandomHorizontalFlip(p=0.5),
        #rotate_crop(degree=15, outputsize=64),
        #torchvision.transforms.RandomRotation(15, interpolation=torchvision.transforms.InterpolationMode.NEAREST),
        #torchvision.transforms.RandomRotation(15, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    
    compose256 = [
        #transforms.ToPILImage(),
        #transforms.RandomHorizontalFlip(p=0.5),
        #rotate_crop(degree=15, outputsize=256),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    
    transform = transforms.Compose(compose)
    transform64 = transforms.Compose(compose64)
    transform256 = transforms.Compose(compose256)
    dataset = FaceDataset(fnames, taglist, transform, transform64, transform256)
    return dataset