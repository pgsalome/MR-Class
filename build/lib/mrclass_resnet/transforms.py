import random
from typing import  Tuple, Union
import numpy as np
import torch
import cv2
from skimage.transform import resize
import itertools
from medpy.filter import otsu
import nibabel as nib


    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if sample is not None:
            image, label,spacing,fn = sample['image'], sample['label'], sample['spacing'], sample['fn']
            # torch.from_numpy(np.float32(image))
            return {'image': torch.from_numpy(np.float32(image)).unsqueeze(dim=0),
                    'label': label,
                    'spacing':spacing,
                    'fn':fn}


class crop_background(object):
    """ base class for background crop """

    def __call__(self,sample):
        if sample is not None:
            image, label,spacing,fn = sample['image'], sample['label'], sample['spacing'], sample['fn']
            threshold = otsu(image)
            output_data = image > threshold
            img = image*output_data
            x,y,z = np.where(img>0)
            new_image = image[x.min():x.max(), y.min():y.max(),:]  
            return {'image': new_image, 'label': label, 'spacing':spacing, 'fn':fn}
        else:
            return None

class check_orientation(object):        
    """ base class for background reorientation """

    def __call__(self,sample):
        if sample is not None:
            image, label,spacing,fn = sample['image'], sample['label'], sample['spacing'], sample['fn']
            mr_image = nib.load(fn)
            x, y, z = nib.aff2axcodes(mr_image.affine)
            if x != 'R':
                image = nib.orientations.flip_axis(image, axis=0)
            if y != 'P':
                image = nib.orientations.flip_axis(image, axis=1)
            if z != 'S':
                image = nib.orientations.flip_axis(image, axis=2)
            return {'image': image, 'label': label, 'spacing':spacing, 'fn':fn}
        else:
            return None
        
class ZscoreNormalization(object):
    """ put data in range of 0 to 1 """
   
    def __call__(self,sample):
        if sample is not None:
            image, label,spacing,fn = sample['image'], sample['label'], sample['spacing'], sample['fn']
            image -= image.mean() 
            image /= image.std() 
            return {'image': image, 'label': label, 'spacing':spacing, 'fn':fn}
        else:
            return None
        
class resize_2Dimage:
    """ Args: img_px_size slices resolution(cubic)
              slice_nr Nr of slices """
    
    def __init__(self,img_px_size):
        self.img_px_size=img_px_size
        
    def __call__(self,sample):
        image, label, spacing, fn = sample['image'], sample['label'], sample['spacing'], sample['fn']
        image_n= cv2.resize(image, (self.img_px_size, self.img_px_size), interpolation=cv2.INTER_CUBIC)
        return {'image': image_n,'label': label, 'spacing':spacing, 'fn':fn}

class resize_3Dimage:
    """ Args: img_px_size slices resolution(cubic)
              slice_nr Nr of slices """
    
    def __init__(self,img_px_size,slice_nr):
        self.img_px_size=img_px_size
        self.slice_nr=slice_nr
    
    def __call__(self,sample):
        image, label, spacing, fn = sample['image'], sample['label'], sample['spacing'], sample['fn']
        s = image.shape
        new_size_x = self.img_px_size
        new_size_y = self.img_px_size
        new_size_z = self.slice_nr
        delta_x = s[0]/new_size_x
        delta_y = s[1]/new_size_y
        delta_z = s[2]/new_size_z
        new_data = np.zeros((new_size_x,new_size_y,new_size_z))
        for x, y, z in itertools.product(range(new_size_x),
                                 range(new_size_y),
                                 range(new_size_z)):
            new_data[x][y][z] = image[int(x*delta_x)][int(y*delta_y)][int(z*delta_z)]
        return {'image': new_data, 'label': label, 'spacing':spacing, 'fn':fn}
    
        

