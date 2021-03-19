# MR-Class

This repository contains a deep learning application for efficient, automatic classification of MR sequences. 
MR-Classifier (MR-Class) is a Convolutional Neural Network (CNN) based method that can distinguish between six classes - T1 weighted (w), constrast-enhanced T1w, T2w, FLAIR, ADC and SWI. It consists of 6 ResNet classifiers, each identifying a single sequence. 
A one-vs-all classification approach was implemented, which allows MR-Class to handle unknown sequences. MR-Class was trained on 29221 images from 320 patients with primary/recurrent high-grade glioma (HGG) and independently validated on 11744 images from 197 recurrent HGG patients. Images not labelled by the CNNs are considered unknown (“other”). Overall accuracy on the independent dataset was 96.2% [95%-CI: 95.82, 96.54], i.e. out of 11744, only 441 MR images were misclassified, mostly due to high anatomical abnormalities or different sorts of MR distortions. 

# Prerequisites

# Installation steps

* Clone or download this repository
* Open a terminal and cd into ```mrclass_resnet``` 
* (Optional) Create a virtual environment for this repository and activate it 
* Run ```python setup.py install```

