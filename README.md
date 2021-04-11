# Pyramid
Pyramid stereo network

This is our blogpost

# Pyramid Stereo Matching Network

This is our blogpost and here we go



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

TELL WHAT WE DID

### Train
REMOVE
As an example, use the following command to train a PSMNet on Scene Flow

```
python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath (your scene flow data folder)\
               --epochs 10 \
               --loadmodel (optional)\
               --savemodel (path for saving model)
```

As another example, use the following command to finetune a PSMNet on KITTI 2015

```
python finetune.py --maxdisp 192 \
                   --model stackhourglass \
                   --datatype 2015 \
                   --datapath (KITTI 2015 training data folder) \
                   --epochs 300 \
                   --loadmodel (pretrained PSMNet) \
                   --savemodel (path for saving model)
```

### Evaluation
REMOVE
```
python submission.py --maxdisp 192 \
                     --model stackhourglass \
                     --KITTI 2015 \
                     --datapath (KITTI 2015 test data folder) \
                     --loadmodel (finetuned PSMNet) \
```

### Pretrained Model


### Test on your own stereo pair
```
python Test_img.py --loadmodel (finetuned PSMNet) --leftimg ./left.png --rightimg ./right.png
```

## Results


```
a=b+c
x+=1
```

### Evaluation of the end-point-error

Here is the epe and we got it this way.

### Disparity maps

Providing images and how we got them

### Evaluation of PSMNet with different settings
REMOVE
<img align="center" src="https://user-images.githubusercontent.com/11732099/37817886-45a12ece-2eb3-11e8-8254-ae92c723b2f6.png">

### Results on KITTI 2015 leaderboard
REMOVE

| Method | D1-all (All) | D1-all (Noc)| Runtime (s) |
|---|---|---|---|
| PSMNet | 2.32 % | 2.14 % | 0.41 |
| [iResNet-i2](https://arxiv.org/abs/1712.01039) | 2.44 % | 2.19 % | 0.12 |
| [GC-Net](https://arxiv.org/abs/1703.04309) | 2.87 % | 2.61 % | 0.90 |
| [MC-CNN](https://github.com/jzbontar/mc-cnn) | 3.89 % | 3.33 % | 67 |

### Qualitative results
REMOVE BUT ADD OUR IMAGES ABOVE
#### Left image
<img align="center" src="http://www.cvlibs.net/datasets/kitti/results/efb9db97938e12a20b9c95ce593f633dd63a2744/image_0/000004_10.png">

#### Predicted disparity
<img align="center" src="http://www.cvlibs.net/datasets/kitti/results/efb9db97938e12a20b9c95ce593f633dd63a2744/result_disp_img_0/000004_10.png">

## Analysis

We performed analysis and we finetuned the model, this showed these interesting things.. And how we did it

Also we did this...

## Conclusion

An interesting project
