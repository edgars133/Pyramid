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

Recent work has shown that depth estimation from a stereo pair of images can be formulated as a supervised learning task to be resolved with convolutional neural networks (CNNs). However, current architectures rely on patch-based Siamese networks, lacking the means to exploit context information for finding correspondence in illposed regions. To tackle this problem, we propose PSMNet, a pyramid stereo matching network consisting of two main modules: spatial pyramid pooling and 3D CNN. The spatial pyramid pooling module takes advantage of the capacity of global context information by aggregating context in different scales and locations to form a cost volume. The 3D CNN learns to regularize cost volume using stacked multiple hourglass networks in conjunction with intermediate supervision.

<img align="center" src="https://user-images.githubusercontent.com/11732099/43501836-1d32897c-958a-11e8-8083-ad41ec26be17.jpg">

## Model Description

### Dependencies

- [Python 3.7](https://www.python.org/downloads/)
- [PyTorch(1.6.0+)](http://pytorch.org)
- torchvision 0.5.0
- [KITTI Stereo](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
- [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

```
Usage of Scene Flow dataset
Download RGB cleanpass images and its disparity for three subset: FlyingThings3D, Driving, and Monkaa.
Put them in the same folder.
And rename the folder as: "driving_frames_cleanpass", "driving_disparity", "monkaa_frames_cleanpass", "monkaa_disparity", "frames_cleanpass", "frames_disparity".
```
### Notice
1. Warning of upsample function in PyTorch 0.4.1+: add "align_corners=True" to upsample functions.
2. Output disparity may be better with multipling by 1.17. Reported from issues [#135](https://github.com/JiaRenChang/PSMNet/issues/135) and [#113](https://github.com/JiaRenChang/PSMNet/issues/113).

### Train
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
You can also see those examples in run.sh.

### Evaluation
Use the following command to evaluate the trained PSMNet on KITTI 2015 test data

```
python submission.py --maxdisp 192 \
                     --model stackhourglass \
                     --KITTI 2015 \
                     --datapath (KITTI 2015 test data folder) \
                     --loadmodel (finetuned PSMNet) \
```

### Pretrained Model
※NOTE: The pretrained model were saved in .tar; however, you don't need to untar it. Use torch.load() to load it.

Update: 2018/9/6 We released the pre-trained KITTI 2012 model.

| KITTI 2015 |  Scene Flow | KITTI 2012|
|---|---|---|
|[Google Drive](https://drive.google.com/file/d/1pHWjmhKMG4ffCrpcsp_MTXMJXhgl3kF9/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1xoqkQ2NXik1TML_FMUTNZJFAHrhLdKZG/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1p4eJ2xDzvQxaqB20A_MmSP9-KORBX1pZ/view?usp=sharing)|

### Test on your own stereo pair
```
python Test_img.py --loadmodel (finetuned PSMNet) --leftimg ./left.png --rightimg ./right.png
```

## Results

### Evaluation of the end-point-error

Here is the epe and we got it this way.

### Disparity maps

Providing images

### Evaluation of PSMNet with different settings
<img align="center" src="https://user-images.githubusercontent.com/11732099/37817886-45a12ece-2eb3-11e8-8254-ae92c723b2f6.png">

※Note that the reported 3-px validation errors were calculated using KITTI's official matlab code, not our code.

### Results on KITTI 2015 leaderboard
[Leaderboard Link](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

| Method | D1-all (All) | D1-all (Noc)| Runtime (s) |
|---|---|---|---|
| PSMNet | 2.32 % | 2.14 % | 0.41 |
| [iResNet-i2](https://arxiv.org/abs/1712.01039) | 2.44 % | 2.19 % | 0.12 |
| [GC-Net](https://arxiv.org/abs/1703.04309) | 2.87 % | 2.61 % | 0.90 |
| [MC-CNN](https://github.com/jzbontar/mc-cnn) | 3.89 % | 3.33 % | 67 |

### Qualitative results
#### Left image
<img align="center" src="http://www.cvlibs.net/datasets/kitti/results/efb9db97938e12a20b9c95ce593f633dd63a2744/image_0/000004_10.png">

#### Predicted disparity
<img align="center" src="http://www.cvlibs.net/datasets/kitti/results/efb9db97938e12a20b9c95ce593f633dd63a2744/result_disp_img_0/000004_10.png">

## Analysis

We performed analysis and we finetuned the model, this ..

## Conclusion

An interesting project
