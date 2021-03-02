import random
from typing import  Tuple, Union
import numpy as np
import torch
import cv2
from skimage.transform import resize
import itertools

class CropBase:
    """ base class for crop transform """

    def __init__(self, out_dim:int, output_size:Union[tuple,int]):
        """ provide the common functionality for RandomCrop2D and RandomCrop3D """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size,)
            for _ in range(out_dim - 1):
                self.output_size += (output_size,)
        else:
            assert len(output_size) == out_dim
            self.output_size = output_size
        self.out_dim = out_dim

    def _get_sample_idxs(self, img:np.ndarray) -> Tuple[int,int,int]:
        """ get the set of indices from which to sample (foreground) """
        mask = np.where(img >= img.mean())  # returns a tuple of length 3
        c = np.random.randint(0, len(mask[0]))  # choose the set of idxs to use
        h, w, d = [m[c] for m in mask]  # pull out the chosen idxs
        return h, w, d
    
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

class rescale_3Dimage:
    
    """Args:spacing_target Desired voxel size."""
       
    def __init__(self,spacing_target):
        self.spacing_target=spacing_target
        
    def __call__(self,sample):
        image, label, old_spacing, fn= sample['image'], sample['label'], sample['spacing'],  sample['fn']
#        image = sitk.GetArrayFromImage(image).astype(float)
        new_spacing=self.spacing_target
        if np.any([[i != j] for i, j in zip(old_spacing, new_spacing)]):
            new_shape = (int(np.round(old_spacing[0]/new_spacing[0]*float(image.shape[0]))),
                         int(np.round(old_spacing[1]/new_spacing[1]*float(image.shape[1]))),
                         int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
            image = resize(image, new_shape, order=3, mode='edge', cval=0, anti_aliasing=False)
            
        return {'image': image, 'label': label, 'spacing':self.spacing_target, 'fn':fn}

    


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
            try:
                new_data[x][y][z] = image[int(x*delta_x)][int(y*delta_y)][int(z*delta_z)]
            except:
                print(fn)
        
    
        return {'image': new_data, 'label': label, 'spacing':spacing, 'fn':fn}
    
        
class RandomCrop3D(CropBase):
    """
    Randomly crop a 3d patch from a (pair of) 3d image

    Args:
        output_size (tuple or int): Desired output size.
            If int, cube crop is made.
    """

    def __init__(self, output_size:Union[tuple,int]):
        super().__init__(3, output_size)

    def __call__(self, sample:Tuple[np.ndarray,np.ndarray]) -> Tuple[np.ndarray,np.ndarray]:
        src, tgt = sample
        *cs, h, w, d = src.shape
        *ct, _, _, _ = tgt.shape
        hh, ww, dd = self.output_size
        max_idxs = (h-hh//2, w-ww//2, d-dd//2)
        min_idxs = (hh//2, ww//2, dd//2)
        s = src[0] if len(cs) > 0 else src  # use the first image to determine sampling if multimodal
        s_idxs = super()._get_sample_idxs(s)
        i, j, k = [i if min_i <= i <= max_i else max_i if i > max_i else min_i
                   for max_i, min_i, i in zip(max_idxs, min_idxs, s_idxs)]
        oh = 0 if hh % 2 == 0 else 1
        ow = 0 if ww % 2 == 0 else 1
        od = 0 if dd % 2 == 0 else 1
        s = src[..., i-hh//2:i+hh//2+oh, j-ww//2:j+ww//2+ow, k-dd//2:k+dd//2+od]
        t = tgt[..., i-hh//2:i+hh//2+oh, j-ww//2:j+ww//2+ow, k-dd//2:k+dd//2+od]
        if len(cs) == 0: s = s[np.newaxis,...]  # add channel axis if empty
        if len(ct) == 0: t = t[np.newaxis,...]
        return s, t

class RandomSlice:
    """
    take a random 2d slice from an image given a sample axis

    Args:
        axis (int): axis on which to take a slice
        div (float): divide the mean by this value in the calculation of mask
            the higher this value, the more background will be "valid"
    """

    def __init__(self, axis:int=0, div:float=2):
        assert 0 <= axis <= 2
        self.axis = axis
        self.div = div

    def __call__(self, sample:Tuple[np.ndarray,np.ndarray]) -> Tuple[np.ndarray,np.ndarray]:
        src, tgt = sample
        *cs, _, _, _ = src.shape
        *ct, _, _, _ = tgt.shape
        s = src[0] if len(cs) > 0 else src  # use the first image to determine sampling if multimodal
        idx = np.random.choice(self._valid_idxs(s)[self.axis])
        s = self._get_slice(src, idx)
        t = self._get_slice(tgt, idx)
        if len(cs) == 0: s = s[np.newaxis,...]  # add channel axis if empty
        if len(ct) == 0: t = t[np.newaxis,...]
        return s, t

    def _get_slice(self, img:np.ndarray, idx:int):
        s = img[...,idx,:,:] if self.axis == 0 else \
            img[...,:,idx,:] if self.axis == 1 else \
            img[...,:,:,idx]
        return s

    def _valid_idxs(self, img:np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """ get the set of indices from which to sample (foreground) """
        mask = np.where(img > img.mean() / self.div)  # returns a tuple of length 3
        h, w, d = [np.arange(np.min(m), np.max(m)+1) for m in mask]  # pull out the valid idx ranges
        return h, w, d
    
class RandomGamma:
    """ apply random gamma transformations to a sample of images """
    def __init__(self, p, tfm_y=False, gamma:float=0., gain:float=0.):
        self.p, self.tfm_y = p, tfm_y
        self.gamma, self.gain = (max(1-gamma,0),1+gamma), (max(1-gain,0),1+gain)

    @staticmethod
    def _make_pos(x): return x.min(), x - x.min()

    def _gamma(self, x, gain, gamma):
        is_pos = torch.all(x >= 0)
        if not is_pos: m, x = self._make_pos(x)
        x = gain * x ** gamma
        if not is_pos: x = x + m
        return x

    def __call__(self, sample:Tuple[torch.Tensor,torch.Tensor]):
        src, tgt = sample
        if random.random() < self.p:
            gamma = random.uniform(self.gamma[0], self.gamma[1])
            gain = random.uniform(self.gain[0], self.gain[1])
            src = self._gamma(src, gain, gamma)
            if self.tfm_y: tgt = self._gamma(tgt, gain, gamma)
        return src, tgt
    
#class N4bias:
#    def __call__(self,sample):
#        image, label, spacing= sample['image'], sample['label'], sample['spacing']
#        maskImage = sitk.OtsuThreshold( image, 0, 1, 200 )
#        inputImage = sitk.Cast( image, sitk.sitkFloat32 )
#        corrector = sitk.N4BiasFieldCorrectionImageFilter();
#        #numberFittingLevels = 4
#        image = corrector.Execute( inputImage, maskImage )
#        return {'image': image, 'label': label, 'spacing':spacing}
#        