# -*- coding: utf-8 -*-

from torchvision import transforms
import torch 
from mrclass_resnet.transforms import ZscoreNormalization, ToTensor,  resize_2Dimage
from mrclass_resnet.utils import load_checkpoint
from mrclass_resnet.MRClassiferDataset import MRClassifierDataset
from torch.utils.data import DataLoader
from collections import defaultdict
import glob

def test(config):
    
    CP_DIR =  config['checkpoint_dir']
    
    checkpoints = {'T1': CP_DIR + '/T1_other.pth',
                   'T2': CP_DIR +'/T2_other.pth',
                   'FLAIR': CP_DIR+'/FLAIR_other.pth',
                   'SWI': CP_DIR+'/SWI_other.pth',
                   'ADC': CP_DIR+'/ADC_other.pth',
                   }
    sub_checkpoints = { 'T1': CP_DIR  + '/T1_T1KM.pth'}
    
    root_dir =   config['root_dir']
    for_inference = glob.glob(root_dir+'/*/*.nii.gz')
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # define transforms
    data_transforms = transforms.Compose(
                    [resize_2Dimage(224),
                    ZscoreNormalization(),
                    ToTensor()])
                    
    # iteration through the different models
    labeled = defaultdict(list)
    for cl in checkpoints.keys():
        model,class_names,scan = load_checkpoint(checkpoints[cl])
        class_names[1] = '@lL'
       
        test_dataset = MRClassifierDataset(list_images = for_inference, transform = data_transforms, class_names = class_names,
                                           scan = scan)
        test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=8)
        for step, data in enumerate(test_dataloader):
            inputs = data['image']
            img_name = data['fn']
            actual_label = data['label']
            inputs = inputs.to(device)
            output = model(inputs)
            prob = output.data.cpu().numpy()
            actRange = abs(prob[0][0])+abs(prob[0][1])
            index = output.data.cpu().numpy().argmax()
            if index == 0:
                labeled[img_name[0]].append([cl,actRange,img_name[-1].split('/')[-2].split('_')[0]])
    
    # check double classification and compare the activation value of class 0
    labeled_correct = defaultdict(list)
    labeled_wrong = defaultdict(list)
    for key in labeled.keys():
        r = 0
        j = 0
        for i in range(len(labeled[key])-1):
            if labeled[key][i][1] > r:
                r = labeled[key][i][1]
                j = i
        for i in range(len(labeled[key])-1):
            if i !=j:
                labeled[key].pop(i)
        predicted_label = labeled[key][0][0]
        true_label = labeled[key][0][2]
        if predicted_label in true_label:
           
            labeled_correct[key] = labeled[key]
        else:
            labeled_wrong[key]= labeled[key]
    
    # check for the unlabeled images
    not_labeled = list(set(for_inference) - set(list(labeled.keys())))
    for img in not_labeled:
        cl = img.split('/')[-2].split('_')[0]
        if cl ==  'other':
            labeled_correct[img].append([cl,'NA',cl])
        else:
            labeled_wrong[img].append(['other','NA',cl])
        
    
    # prepare for subclassification   
    labeled_images = defaultdict(list)
    for key in labeled_correct.keys():
        labeled_images[labeled_correct[key][0][0]].append(key)
        
    # subclassification        
    labeled_subcorrect = defaultdict(list)
    labeled_subwrong = defaultdict(list)
    for cl in sub_checkpoints.keys():
        model,class_names,scan = load_checkpoint(sub_checkpoints[cl])
        test_dataset = MRClassifierDataset(list_images = labeled_images[cl], 
                                           transform = data_transforms,
                                           class_names = class_names, scan = scan, 
                                           subclasses = True)
        test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=8)
        for step, data in enumerate(test_dataloader):
            inputs = data['image']
            img_name = data['fn']
            actual_label = data['label']
            inputs = inputs.to(device)
            output = model(inputs)
            prob = output.data.cpu().numpy()
            actRange = abs(prob[0][0])+abs(prob[0][1])
            index = output.data.cpu().numpy().argmax()
            actual_label = img_name[0].split('/')[-2].split('_')[-1]
            if index == 1:
                c = 'KM'
            else:
                c = ''
            if index == class_names.index(actual_label):

                labeled_subcorrect[img_name[0]].append([cl+c,actRange,actual_label])
            else:
                labeled_subwrong[img_name[0]].append([cl+c,actRange,actual_label])
    
    for key in labeled_subcorrect.keys():
        labeled_correct[key] = labeled_subcorrect[key]
    for key in labeled_subwrong.keys():
        labeled_wrong[key] = labeled_subwrong[key]
        
    # calculate accuracy    
    T1_acc =  sum(value[0][2] == 'T1' for value in labeled_correct.values()) / \
                (sum(value[0][2] == 'T1' for value in labeled_correct.values()) + \
                sum(value[0][2] == 'T1' for value in labeled_wrong.values()))
    T1KM_acc =  sum(value[0][2] == 'T1KM' for value in labeled_correct.values()) / \
                (sum(value[0][2] == 'T1KM' for value in labeled_correct.values()) + \
                sum(value[0][2] == 'T1KM' for value in labeled_wrong.values()))
    T2_acc =  sum(value[0][2] == 'T2' for value in labeled_correct.values()) / \
                (sum(value[0][2] == 'T2' for value in labeled_correct.values()) + \
                sum(value[0][2] == 'T2' for value in labeled_wrong.values()))
    FL_acc =  sum(value[0][2] == 'FLAIR' for value in labeled_correct.values()) / \
                (sum(value[0][2] == 'FLAIR' for value in labeled_correct.values()) + \
                sum(value[0][2] == 'FLAIR' for value in labeled_wrong.values()))                
    ADC_acc =  sum(value[0][2] == 'ADC' for value in labeled_correct.values()) / \
                (sum(value[0][2] == 'ADC' for value in labeled_correct.values()) + \
                sum(value[0][2] == 'ADC' for value in labeled_wrong.values()))  
    SWI_acc =  sum(value[0][2] == 'SWI' for value in labeled_correct.values()) / \
                (sum(value[0][2] == 'SWI' for value in labeled_correct.values()) + \
                sum(value[0][2] == 'SWI' for value in labeled_wrong.values()))   
    total_acc = len(labeled_correct)/(len(labeled_correct)+len(labeled_wrong))
    

info = """\
{'-'*40}
# Accuracies of the different MR sequence models
# T1: {T1_acc}
# T1-Contrast agent: {T1KM_acc}
# T2: {T2_acc}
# FLAIR: {FL_acc}
# ADC: {ADC}
# SWI: {SWI}
# '-'*40
# MR-Class accuracy: {total_acc}
{'-'*40}
"""

print(info)
    