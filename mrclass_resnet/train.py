
from __future__ import print_function, division

import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
import matplotlib.pyplot as plt
from mrclass_resnet.transforms import ZscoreNormalization, ToTensor, resize_3Dimage, resize_2Dimage
from mrclass_resnet.MRClassiferDataset import MRClassifierDataset
from torch.utils.data import DataLoader
from mrclass_resnet.utils import  get_dataset_sizes, get_train_images
from mrclass_resnet.train_model import train_model
import math
import numpy as np

def train(config):
    
    plt.ion()   # interactive mode
    
    # retrieve variables
    root_dir =  config['root_dir'] 
    class_names = config['class_names']
    if len(class_names) == 1:
        class_names = [class_names[0],'@lL']
    num_classes = len(class_names)
    spatial_size = config['spatial_size']
    nr_slices = config['nr_slices']
    training_split = config['training_split']
    scan =  config['scan'] # for 2d; 2:axial, 1:coronal, 0:sagittal, 3:scan aquisation plane
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    in_channels = config['in_channels']
    lr = config['learning_rate']
    momentum = config['momentum']
    step_size = config['step_size'] 
    gamma = config['gamma']
    num_workers = config['num_workers']
    run_3d = config['run_3d']
    checkpoint_dir = config['checkpoint_dir'] 
    data_aug = config['data_aug']
    show_results = config['show_results']
    num_images = config['num_images']
    remove_corrupt = config['remove_corrupt'] 
    subclasses = config['subclasses'] 
    onevsall = config['onevsall'] 
    parentclass = config['parentclass']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        print('Warning, training on cpu')
    
    # retrieve scans
    if onevsall:
        list_images = glob.glob(root_dir + '/*/*.nii*') 
    else:
        if subclasses and parentclass: 
            list_images = [x for x in glob.glob(root_dir + '/*/*') \
                           if x.split('/')[-2].split('_')[0] in class_names ]
        elif subclasses and not parentclass:
            list_images = [x for x in glob.glob(root_dir + '/*/*') \
                           if x.split('/')[-2].split('_')[-1] in class_names ]
        else:
            list_images = [x for x in glob.glob(root_dir + '/*/*') \
                           if x.split('/')[-2] in class_names ]        
        
    
    # stratify scans based on classes.
    train_class = get_dataset_sizes(list_images,class_names,
                                    training_split,subclasses,parentclass)
    train_images = get_train_images(list_images,train_class,class_names,
                                    subclasses,parentclass)
    test_images = list(set(list_images) - set(train_images))
    
    #define transforms
    if run_3d:
        data_transforms = transforms.Compose(
                
                        [
                        resize_3Dimage(spatial_size,nr_slices),
                         ZscoreNormalization(),
                         ToTensor()])
    else:
        data_transforms = transforms.Compose(
                
                        [
                        resize_2Dimage(spatial_size),
                         ZscoreNormalization(),
                         ToTensor()])    
    if data_aug:
        from mrclass_resnet.data_augmentation import get_aug_pipeline
        aug_train = get_aug_pipeline()
    else:
        aug_train = None
    
    #define datasets
    train_dataset = MRClassifierDataset(list_images = train_images,augmentations=aug_train,
                                                transform=data_transforms, class_names=class_names, 
                                                run_3d=run_3d, scan = scan, remove_corrupt=remove_corrupt,
                                                subclasses=subclasses, parentclass=parentclass,
                                                spatial_size=spatial_size,nr_slices=nr_slices)
    test_dataset = MRClassifierDataset(list_images = test_images,augmentations=aug_train,
                                                transform=data_transforms, class_names=class_names, 
                                                run_3d=run_3d, scan = scan, remove_corrupt=remove_corrupt,
                                                subclasses=subclasses, parentclass=parentclass,
                                                spatial_size=spatial_size,nr_slices=nr_slices)
    
    image_datasets = {"train":train_dataset,
                      "val":test_dataset 
                      }
    # define dataloaders
    dataloaders = {x:DataLoader(image_datasets[x], batch_size,
                                                 shuffle=True, num_workers=num_workers)
                  for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


    if run_3d:
        model_ft = models.video.r3d_18(pretrained=False)
        model_ft.stem[0] = nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
    else:

        model_ft = models.resnet18(pretrained=False)
        model_ft.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model_ft = model_ft.to(device)
    
    # define weights for loss function
    high = max(train_class)
    weights = [0 for x in range(num_classes)]
    for i in range(num_classes):
        weights[i] = math.floor(high / train_class[i])
        
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr = lr, momentum=momentum)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)
    
    model_ft, best_acc = train_model(model_ft,dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs,device,dataset_sizes)
    
    if show_results:
        from mrclass_resnet.utils import visualize_model 
        visualize_model(model_ft,dataloaders,class_names,device,num_images,checkpoint_dir)
    
    #store checkpoint
    checkpoint = {'model': model_ft,
              'state_dict': model_ft.state_dict(),
              'optimizer' : optimizer_ft.state_dict(),
              'class_names': class_names,
              'scan_plane':scan}
    
    if onevsall:
        class_names[-1] = 'other'
    c_name = "_".join(class_names)
    torch.save(checkpoint, checkpoint_dir+'/'+c_name+'_'+str(np.round(best_acc.cpu().numpy(),4))+'.pth')
