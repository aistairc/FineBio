# I3D Feature Calculation

This repository contains the codes to extract I3D features by using the implementation of the paper "[Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750.pdf)" 


## Requirements
The code has been tested with Python 3.10, PyTorch 1.12.1 and CUDA 11.3.
```Shell
conda create --name i3d -y python==3.10
conda activate i3d
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install opencv-python tqdm scipy
```

## Data Preparation
#### RGB
+ Download the FineBio videos.

#### Flow
+ First get the optical flow by using [RAFT](../RAFT).
+ Then the flow folder should look like
    ```
    The flow folder
    └───P01/
    │    └───P01_01_01/
    │    │	 └───flow_000001_x.png
    │    │	 └───flow_000001_y.png
    │    │   └───...
    │    └───...
    |
    └───P02/
    │
    │   ...
    ```


## Pretrained Models
Download models pretrained on imagenet and kinetics (`rgb_imagenet.pt` and `flow_imagenet.pt`) from [this repository](https://github.com/piergiaj/pytorch-i3d). 


## Feature Extraction
You can extract features by using `extract_feature.py`
#### RGB
```Shell
python extract_feature.py \
-mode rgb \
-data_dir [/path/to/FineBio/videos] \
-model [/path/to/rgb_imagenet.pt] \
-max_len 640 \
-view_id T0 \
-save_dir [path/to/output/folder]
```

#### Flow
```Shell
python extract_feature.py \
-mode flow \
-data_dir [path/to/flow/images] \
-model [/path/to/flow_imagenet.pt] \
-max_len 640 \
-view_id T0 \
-save_dir [path/to/output/folder]
```

For features of Step Segmentation, set the other arguments as below.
```Shell
-width 21 -stride 1 -symmetry
```
For features of Atomic Operation Detection, set the other arguments as below.
```Shell
-width 16 -stride 4
```

`-max_len` represents a maximum length of the image. If the length of video frame is over `-max_len`, the longer side of frame will be resized into `-max_len`.  
`-view_id` indicates the id of the viewpoint of the videos. Set T0 for first-person videos, and the last two letters in the video file name (e.g., T1, T2, ...) for third-person videos.  
`-width` and `-stride` mean a length of a clip and a stride between clips given to the I3D model.


Since this repository is built based on [pytorch-i3d](https://github.com/piergiaj/pytorch-i3d), please refer to the original repository if you have further questions.