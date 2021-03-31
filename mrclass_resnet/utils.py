import os

import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import yaml
import torch
import pandas as pd
from torch.autograd import Variable
import nibabel as nib
from mrclass_resnet.DicomConvert import DicomConverters
import random


def get_dataset_sizes(list_images,class_names,
                      training_split,subclasses,parentclass):  
    count = [0] * len(class_names)   
    for i in range(len(class_names)):                                                   
        for item in list_images:
            if subclasses and parentclass: 
                item_class = item.split('/')[-2].split('_')[0] 
            elif subclasses and not parentclass:
                item_class = item.split('/')[-2].split('_')[-1] 
            else:
                item_class = item.split('/')[-2]
            if item_class == class_names[i]:                                                       
                count[i] += 1
    if class_names[-1] == '@lL':
        count[-1] = len(list_images) - count[0]
    train_class = []
    for i in range(len(class_names)):
        train_class.append( math.ceil((training_split) * count[i]))
    
    return train_class


def get_images(list_images,direction):
    l=[]
    for image in list_images:
        l.append(image)        
    return l

def get_label(name, class_names):
    if class_names[-1] == '@lL':
        if class_names[0] == name:
            label = 0
        else:
            label = 1
    else:
        for i in range(len(class_names)):
            if class_names[i] == name:
                label = i
    return label


def check_size(list_images):
    l=[]
    for image in list_images:
        size=os.stat(image).st_size
        if size > 4000000:
            l.append(image)
    return l

def show_batchedSlices(sample_batched,nr_of_slices):
    """Show labeled slices"""
    images_batch = sample_batched['image']
    for i in range(images_batch.size(0)):
        show_2dimage(images_batch[i,:,:,])
        
        

def show_2dimage(i):
    """Show labeled slices"""
    image=torch.squeeze(i)
    plt.figure()
    plt.imshow(image,cmap='gray')
    # pause a bit so that plots are updated
    plt.pause(0.001)
    plt.show()           
         

def show_slices(i,Scan,nr_of_slices=4,label=[1,0],spacing=(1.5,1.5,1.5)):
    """Show labeled slices"""
    image=torch.squeeze(i)
    plt.figure()
    for i in range(nr_of_slices):

        ms=math.ceil(image.shape[0]/2)
        slice=ndimage.rotate(image[ms+i,:,:],180)
        ax = plt.subplot(1, nr_of_slices, i + 1)
        ax.set_title('Sample #{}'.format(Scan[:-7]))
        ax.axis('off')
        plt.imshow(slice,cmap='gray')
        # pause a bit so that plots are updated
        plt.pause(0.001)
    plt.show()    
        
def subfiles(f, modalities, join=True, prefix=None, suffix=None, sort=True):
    imageList=[]
    for j, m in enumerate(modalities):
        folder= f +'/'+m
        if join:
            l = os.path.join
        else:
            l = lambda x, y: y
        res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
               and (prefix is None or i.startswith(prefix))
               and (suffix is None or i.endswith(suffix))]
        if sort:
            res.sort()
            
        imageList+=res
    return imageList

def glob_nii(dir):
    """ return a sorted list of nifti files for a given directory """
    fns = sorted(glob(os.path.join(dir, '*.nii.gz')))
    return fns

def split_filename(filepath):
    """ split a filepath into the full path, filename, and extension (works with .nii.gz) """
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext

def save_checkpoint(save_file_path, epoch, model_state_dict, optimizer_state_dict):
    states = {'epoch': epoch+1, 'state_dict': model_state_dict, 'optimizer':  optimizer_state_dict}
    torch.save(states, save_file_path)

def cleanup_checkpoint_dir(checkpoint_dir,checkpoints_num_keep):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'save_*.pth'))
    checkpoint_files.sort()
    if len(checkpoint_files) > checkpoints_num_keep:
        os.remove(checkpoint_files[0])


def load_checkpoint(filepath):
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    checkpoint = torch.load(filepath, map_location=map_location)
    
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model,checkpoint['class_names'],checkpoint['scan_plane']

def image_loader(img_name,data_transforms):
    """load image, returns cuda tensor"""

    converter = DicomConverters(img_name, ext='.nii.gz')
    niftiImage=converter.dcm2niix_converter(compress=True)
    image = nib.load(niftiImage).get_data()
    if len(image.shape)>3:
        print('4D images are not supported;{} is ignored'.format(img_name))
        return  None
    image=extract_middleSlice(image)
    sample = {'image': image, 'label': np.array(0), 'spacing': 0}
    sample = data_transforms(sample)
    image = Variable(sample['image'], requires_grad=True)
    image = image.unsqueeze(0) 
    return image.cuda()  #assumes that you're using GPU
     
def extract_middleSlice(image, scan):
    x,y,z = image.shape
    s = smallest(x,y,z)
    if (s==z and scan == 3) or scan == 2:
        ms= math.ceil(image.shape[2]/2)-1
        return image[:, :, ms].astype('float32')
    elif (s==y and scan == 3) or scan == 1:
        ms=math.ceil(image.shape[1]/2)-1
        return image[:, ms, :].astype('float32')
    else:
        ms= math.ceil(image.shape[0]/2)-1
        return image[ms, :, :].astype('float32')
    
def smallest(num1, num2, num3):
    if (num1 < num2) and (num1 < num3):
        smallest_num = num1
    elif (num2 < num1) and (num2 < num3):
        smallest_num = num2
    else:
        smallest_num = num3
    return smallest_num

def imshow_batch(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(model,dataloaders,class_names,device, num_images=6,checkpoint_dir = None):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for steps,sample in enumerate(dataloaders['val']):
            
            inputs=sample['image']
            labels=sample['label']
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                if class_names[preds[j]] == '@lL':
                    c = 'other'
                else:
                    c = class_names[preds[j]]
                ax.set_title('predicted: {}'.format(c))
                imshow_batch(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    if checkpoint_dir is not None:
                        fig.savefig(checkpoint_dir+'/model_'+str(num_images)+'.png')
                    return
            model.train(mode=was_training)
            
def plot_cm(csv):
    df = pd.read_csv(csv,index_col=0).astype(float)
    df_conf_norm = df/ df.sum(axis=1)
    
    plt.matshow(df_conf_norm,cmap=plt.cm.Blues)
    cbar = plt.colorbar()
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(24)
    for (i, j), z in np.ndenumerate(df_conf_norm):
        if z > 0.8:
            plt.text(j, i, '{:0.3f}'.format(z), ha='center', va='center',fontsize=24,color='white')
        else:
            plt.text(j, i, '{:0.3f}'.format(z), ha='center', va='center',fontsize=24)
            
    tick_marks = np.arange(len(df_conf_norm.columns))
    plt.xticks(tick_marks, df_conf_norm.columns, rotation=45,fontsize=30)
    plt.yticks(tick_marks, df_conf_norm.index,fontsize=30)
    plt.ylabel(df_conf_norm.index.name)
    plt.xlabel(df_conf_norm.columns.name)
    
def load_config(config_file):
    return yaml.safe_load(open(config_file, 'r'))

def get_train_images(list_images,train_class,
                     class_names,subclasses,parentclass):
    
    count_train = [0 for x in range(len(class_names))]
    train_images = [] 
    for l in range(len(class_names)):
        for x,j in enumerate(list_images):
            if subclasses and parentclass: 
                item_class = j.split('/')[-2].split('_')[0] 
            elif subclasses and not parentclass:
                item_class = j.split('/')[-2].split('_')[-1] 
            else:
                item_class = j.split('/')[-2]
            if class_names[l] == item_class :
                if count_train[l]  < train_class[l]:
                    train_images.append(j)
                    count_train[l]  +=1
    if class_names[-1] == '@lL':
        
        rest = list(set(x for x in list_images if class_names[0] not in x) - set(train_images))
        train_images = train_images + random.sample(rest, train_class[-1])
    return train_images  
