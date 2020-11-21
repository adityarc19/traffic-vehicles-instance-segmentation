# Real time traffic vehicles instance segmentation with YOLACT++

**This work is inspired from [Immersive Limit](https://www.immersivelimit.com/tutorials/yolact-with-google-colab).**

![im3][log]

[log]: https://github.com/adityarc19/yolact-plus/blob/master/output.gif


YOLACT (You Only Look At Coefficients) is a simple, fully convolutional model for real-time instance segmentation. The latest version of it which was released on 16th Decemeber 2019, is called YOLACT++. It currently supports two backbone networks : Resnet50-FPN and Resnet101-FPN. I have used Resnet50 version for this project. It is originally trained on a very sophisticated graphics card, i.e., Titan Xp. That is the reason why I have not trained the model on my local computer and just used the pre-trained weights provided by the authors of this technology.

Check out YOLACT and YOLACT++ papers: 

YOLACT : https://arxiv.org/abs/1904.02689

YOLACT++ : https://arxiv.org/abs/1912.06218

Also, check out the original github repo of YOLACT [here](https://github.com/dbolya/yolact).

## Initial setup

Run the following commands for the initial setup:

```
# Cython needs to be installed before pycocotools
!pip install cython
!pip install opencv-python pillow pycocotools matplotlib
```

```
# Downgrade torch to accommodate DCNv2
!pip install torchvision==0.5.0
!pip install torch==1.4.0
```

```
# Clone the yolact repo
!git clone https://github.com/dbolya/yolact.git
```

```
# The DCNv2 external library is needed for this to work
%cd /content/yolact/external/DCNv2

# Build DCNv2
!python setup.py build develop
```

## Download the pre-trained weights

In order to run inference, we'll need some pre-trained weights. The creator of the GitHub repo shared them on Google Drive. We're going to use a [helpful tool](https://github.com/chentinghao/download_google_drive) made by [chentinghao](https://github.com/chentinghao) to easily access the Drive file from Colab.

```
# Make sure we're in the top folder
%cd /content

# Clone the repo
!git clone https://github.com/chentinghao/download_google_drive.git

# Create a new directory for the pre-trained weights
!mkdir -p /content/yolact/weights

# Download the pre-trained weights file
!python ./download_google_drive/download_gdrive.py 1ZPu1YR2UzGHQD0o1rEqy-j5bmEm3lbyP ./yolact/weights/yolact_plus_resnet50_54_800000.pth
```

## Other regular data science libs 

```
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
```

## Training 

I have used a video of cars and buses going by on a busy street. It also has pedestrians walking by the streets in a crowded fashion. The video is attached [here](https://github.com/adityarc19/yolact-plus/blob/master/taxi.mp4).

After the training process was over, I got a pretty good result of an instance segmented video that is attached [here](https://github.com/adityarc19/yolact-plus/blob/master/taxi_output.mp4).









