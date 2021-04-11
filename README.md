# Pyramid
Pyramid stereo network

This is our blogpost

Contents:
Intro
Describe model
-pretrained sceneflow model
-three datasets monkaa etc consisting of right,left and disparity image

-computing test loss, on  a single dataset giving epe.
-we found epe on pretrained models

-we compared images 
-then we finetuned the model

Results and how we did it
-epe
-disparity maps

Analysis
-we finetuned model, results and how

Conclusion

# Pyramid Stereo Matching Network

This is our blogpost and here we go
Here is the PSMNet repository made by the authors.

https://github.com/JiaRenChang/PSMNet

## Contents

1. [Introduction](#introduction)
2. [Model Description](#model)
3. [Results](#results)
4. [Analysis of model](#analysis)
5. [Conclusion](#conclusion)

## Introduction

DO INTRO
''Recent work has shown that depth estimation from a stereo pair of images can be formulated as a supervised learning task to be resolved with convolutional neural networks (CNNs). However, current architectures rely on patch-based Siamese networks, lacking the means to exploit context information for finding correspondence in illposed regions. To tackle this problem, we propose PSMNet, a pyramid stereo matching network consisting of two main modules: spatial pyramid pooling and 3D CNN. The spatial pyramid pooling module takes advantage of the capacity of global context information by aggregating context in different scales and locations to form a cost volume. The 3D CNN learns to regularize cost volume using stacked multiple hourglass networks in conjunction with intermediate supervision.''



## Model Description
DESCRIBE MODEL BRIEFLY
<img align="center" src="https://user-images.githubusercontent.com/11732099/43501836-1d32897c-958a-11e8-8083-ad41ec26be17.jpg">

The model uses sceneflow dataset which concists of 3 subsets - 'Driving', 'Flying 3Dthings' and 'Monkaa'. The model which is pretrained on sceneflow dataset was used to measure the end-point-error. Afterwards the model was finetuned on KITTI dataset and results were compared.

We tried to reproduce the results of end-point-error by evaluating the pretrained model on sample 'Monkaa' dataset and 'Driving' dataset separately. Afterwards, ablation study was performed to see what are the consequences of finetuning the pretrained model. 


```
python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath (your scene flow data folder)\
               --epochs 10 \
               --loadmodel (optional)\
               --savemodel (path for saving model)
```

```
python finetune.py --maxdisp 192 \
                   --model stackhourglass \
                   --datatype 2015 \
                   --datapath (KITTI 2015 training data folder) \
                   --epochs 300 \
                   --loadmodel (pretrained PSMNet) \
                   --savemodel (path for saving model)
```

```
python Test_img.py --loadmodel (finetuned PSMNet) --leftimg ./left.png --rightimg ./right.png
```

## Results


```
a=b+c
x+=1
```

### Evaluation of the end-point-error

End-point-error is defined as the test loss. This numeric in the original paper was obtained as 1.09, however, larger values were found during the reproduction study. On 'Monkaa' sample subset the error was as large as 10.

[INFO ABOUT EPE PROBLEM]






## Analysis

As a second step in our reproducibility project, it has been decided to use the Sceneflow pretrained model and finetune it on KITTI 2015, hence comparing the disparity images and find qualitative differences in the images. For the finetune, Google Colab was chosen. Since the RAM memory was limited, the batch size was reduced from 12 to 4 and the number of epochs was set to 300 epochs. Unfortunately the runtime of the VM in Google Colab is limited (12 hours) hence we managed to finetune the model for 186 epochs. Nevertheless, this was enough to get disparity images.
To compare the pretrained model and the finetuned model, we decided to test the models on two images, one from Driving dataset and one from Monkaa dataset (both subsets of Sceneflow dataset).

#### Left image

<p float="left">
  <img src="0401_s.png" width="100" />
  <img src="0401_finetune.png" width="100" /> 
</p>



## Conclusion

An interesting project
