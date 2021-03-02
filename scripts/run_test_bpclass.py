# -*- coding: utf-8 -*-

from torchvision import transforms
import torch 
from mrclass_resnet.transforms import ZscoreNormalization, ToTensor,  resize_2Dimage
from mrclass_resnet.utils import load_checkpoint, pop_wrong
from mrclass_resnet.MRClassiferDataset import MRClassifierDataset
from torch.utils.data import DataLoader
from collections import defaultdict
import glob

PARENT_DIR =  '/home/e210/Python/mrclass_resnet/mrclass_resnet'

checkpoints = {'abd-pel':PARENT_DIR+'/checkpoints/bp-class/abd-pel_other_09402.pth',
                'lung':PARENT_DIR+'/checkpoints/bp-class/lung_other_09957.pth',
                'hnc':PARENT_DIR+'/checkpoints/bp-class/hnc_other_09974.pth',
                'wb':PARENT_DIR+'/checkpoints/bp-class/wb_other_09775.pth'
                   }

sub_checkpoints = {
                'wb':PARENT_DIR + '/checkpoints/bp-class/wb_wbh0_09213.pth',
                'hnc': PARENT_DIR + '/checkpoints/bp-class/brain_hnc_0.9425.pth',
}

root_dir =   '/media/e210/Samsung_T5/test'
for_inference = glob.glob(root_dir+'/*/*.nii.gz')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose(
                [resize_2Dimage(224),
                ZscoreNormalization(),
                ToTensor()])

labeled = defaultdict(list)
other  = defaultdict(list)
for cl in checkpoints.keys():
    model,class_names,scan = load_checkpoint(checkpoints[cl])
   
    test_dataset = MRClassifierDataset(list_images = for_inference, transform = data_transforms, class_names = class_names,
                                       scan = scan, remove_corrupt = True)
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
            labeled[img_name[0]].append([cl,actRange,img_name[-1].split('/')[-2]])

ll = labeled.copy()
correct = 0
wrong = []
w = []
labeled_correct = defaultdict(list)

for key in labeled.keys():
    r = 0
    for i in range(0,len(labeled[key])):
        if labeled[key][i][1] > r:
            r = labeled[key][i][1]
            j = i
    pop_wrong(labeled[key],j)
    predicted_label = labeled[key][0][0]
    true_label = labeled[key][0][2]
    if predicted_label in true_label:
        correct+=1
        labeled_correct[key] = labeled[key]
    else:
        print(key)
        wrong.append(key)
        
labeled_images = defaultdict(list)
for key in labeled_correct.keys():
    labeled_images[labeled_correct[key][0][0]].append(key)

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
        
        if index == class_names.index(actual_label):
            labeled_subcorrect[img_name[0]].append([cl,actRange,actual_label])
        else:
            labeled_subwrong[img_name[0]].append([cl,actRange,class_names[index]])

for key in labeled.keys():
    r = 0
    for i in range(0,len(labeled[key])):
        if labeled[key][i][1] > r:
            r = labeled[key][i][1]
            j = i
    pop_wrong(labeled[key],j)
    predicted_label = labeled[key][0][0]
    true_label = labeled[key][0][2]
    if predicted_label in true_label:
        correct+=1
        labeled_correct[key] = labeled[key]
    else:
        print(key)
        wrong +=1   
