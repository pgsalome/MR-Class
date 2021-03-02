  
from mrclass_resnet.utils import get_label,extract_middleSlice
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import os

class MRClassifierDataset(Dataset):

    def __init__(self,list_images='', transform=None, augmentations=None, 
                 class_names = '', run_3d = False, scan = 0, 
                 remove_corrupt = True, subclasses = False,
                 parentclass = False, inference = False, spatial_size=224, 
                 nr_slices = 50):
 
        self.transform = transform
        self.list_images = list_images
        self.class_names = class_names
        self.augmentations = augmentations
        self.run_3d = run_3d
        self.scan = scan
        self.remove_corrupt = remove_corrupt
        self.subclasses = subclasses
        self.parentclass = parentclass
        self.nr_slices = nr_slices
        self.spatial_size = spatial_size
        
    def __len__(self):
        return len(self.list_images)
    
    def get_random(self):
        
        if self.run_3d:
            image = np.random.randn(self.spatial_size, self.spatial_size,self.nr_slices).astype('f')
        else:
            image = np.random.randn(self.spatial_size, self.spatial_size).astype('f')
        class_cat = 'random'
        return image, class_cat
    
    def __getitem__(self, idx):
        
        #modify the collate_fn from the dataloader so that it filters out None elements.
        img_name = self.list_images[idx]

        try:
            image = nib.load(img_name).get_data()
            if self.subclasses and self.parentclass:
                class_cat = img_name.split('/')[-2].split('_')[0]
            elif self.subclasses and not self.parentclass:
                class_cat = img_name.split('/')[-2].split('_')[-1]
            else:
                class_cat = img_name.split('/')[-2]
        except:
            print ('error loading {0}'.format(img_name))
            if self.remove_corrupt and os.path.isfile(img_name):
                os.remove(img_name)
            image, class_cat = self.get_random()
            
        if len(image.shape)>3:
            image = image[:,:,:,0]
            #print('4D images are not supported;{} is ignored'.format(img_name))
            
        if not self.run_3d:
            try:
                image = extract_middleSlice(image, self.scan)                 
            except:
                print ('error loading {0}'.format(img_name))
                if self.remove_corrupt and os.path.isfile(img_name):
                    os.remove(img_name)
                image, class_cat = self.get_random()
        spacing = image.shape
        
        label = get_label(class_cat, self.class_names)
        if self.augmentations is not None:
            image = self.augmentations.augment_image(image)
        sample = {'image': image, 'label': np.array(label), 'spacing': spacing, 'fn': img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample
