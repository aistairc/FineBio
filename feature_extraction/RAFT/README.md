# Optical Flow Calculation using RAFT

This repository contains the codes to calculate optical flow by using the implementation of the paper "[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)". 


## Requirements
The code has been tested with Python 3.10, PyTorch 1.12.1 and CUDA 11.3.
```Shell
conda create --name raft -y python==3.10
conda activate raft
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install opencv-python tqdm scipy
```


## Data Preparation
+ Download the FineBio videos.


## Pretrained Models
+ Download the pretrained RAFT models by running
    ```Shell
    ./download_models.sh
    ```
    or from [google drive](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing)
+ We use `raft-sintel.pth` for the flow calculation of the FineBio dataset.


## Flow Calculation
**NOTE:** You need approximately 110GB to save flow images for all the videos from each view point.
You can get flow by using `get_finebio_flow.py`.
```Shell
python get_finebio_flow.py \
--model_path [/path/to/raft-sintel.pth] \
--data_dir [/path/to/FineBio/videos] \
--save_dir [/path/to/output/folder] \
--max_len 640 \
--view_id T0
```
`--max_len` represents a maximum length of the image. If the length of video frame is over `--max_len`, the longer side of frame will be resized into `--max_len`.  
`--view_id` indicates the id of the viewpoint of the videos. Set T0 for first-person videos, and the last two letters in the video file name (e.g., T1, T2, ...) for third-person videos.


Since this repository is built based on [RAFT](https://github.com/princeton-vl/RAFT), please refer to the original repository if you have further questions.

