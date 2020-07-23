# Real time instance segmentation with YOLACT++

**This is a real time instance segmentation task implemented with YOLACT++ and DCNv2 on Google Colab.**

![sample][logo]

[logo]: https://github.com/adityarc19/yolact-plus/blob/master/ezgif.com-crop.gif


YOLACT (You Only Look At Coefficients) is a simple, fully convolutional model for real-time instance segmentation. The latest version of it which was released on 16th Decemeber 2019, is called YOLACT++. It currently supports two backbone networks : Resnet50-FPN and Resnet101-FPN. I have used Resnet50 version for this project. It is originally trained on a very sophisticated graphics card, i.e., Titan Xp. That is the reason why I have not trained the model on my local computer and just used the pre-trained weights provided by the authors of this technology.

Check out YOLACT and YOLACT++ papers: 

YOLACT : https://arxiv.org/abs/1904.02689

YOLACT++ : https://arxiv.org/abs/1912.06218

Also, check out the original github repo of YOLACT [here](https://github.com/dbolya/yolact).

## Initial setup

*Before starting the installation process, note that I have used **Google Colab** for this project. So if you're on some other platform, make the necessary changes accordingly, although I recommend using Colab as it provides a lot easier access to Google Drive which comes in handy when you have heavy files to upload and download.*

Now run the following commands for the initial setup:

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
# Make sure we're in the top folder
%cd /content

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

## Output

Some sample output images after going through the training process are:

![im1][l]

[l]: https://github.com/adityarc19/yolact-plus/blob/master/output%20images/bears.png


![im2][lo]

[lo]: https://github.com/adityarc19/yolact-plus/blob/master/output%20images/plane.png












