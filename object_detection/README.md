# Object Detection benchmark of the FineBio dataset

This repository contains the code for Object Detection benchmark of the FineBio dataset.

## Requirements
The code has been tested with Python 3.8.16, PyTorch 1.12.1 and CUDA 11.3.
```Shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Data Preparation
+ Download `finebio_coco_annotations.zip` and `finebio_object_detection_images.zip` and unzip them in a same folder. 
+ The files should be located as follows:
    ```
    The folder
    └───annotations/
    │    └───v1_train_fpv.json
    │    └───v1_test_fpv.json
    │    └─── ...
    |
    └───images/
    │    └───P01_01_01_000152.jpg
    │    └───P01_01_01_000983.jpg
    │    └─── ...
    ```
+ Write the path of the folder to `data_root` in [deformable-detr-refine-twostage_r50_16xb2-50e_finebio.py](deformable-detr-refine-twostage_r50_16xb2-50e_finebio.py) and [dino-4scale_r50_8xb2-12e_finebio.py](dino-4scale_r50_8xb2-12e_finebio.py).


## Code Preparation
We use [MMDetection](https://github.com/open-mmlab/mmdetection) implementation of [Deformabel DETR](https://arxiv.org/pdf/2010.04159.pdf) and [DINO](https://arxiv.org/pdf/2203.03605.pdf). Since we do not change the major codes in MMDetection, this repository only provides the config files for the FineBio dataset and the codes for our new metrics (AP\_manipulated and AP\_affected). Follow the instructions below to complete the codebase by combining our files with MMDetection .
+ First install MMDetection.   
  ```Shell
  pip install -U openmim
  mim install mmengine
  mim install "mmcv>=2.0.0"

  git clone https://github.com/open-mmlab/mmdetection.git
  cd mmdetection
  pip install -v -e .
  ```
+ Replace an original file in `mmdetection` with our file.  
    + [\__init\__.py](__init__.py) -> [mmdetection/mmdet/evaluation/metrics/\__init\__.py](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/evaluation/metrics/__init__.py)
+ Add our new files in `mmdetection`.
    + [coco_manipulated_metric.py](coco_manipulated_metric.py) -> [mmdetection/mmdet/evaluation/metrics](https://github.com/open-mmlab/mmdetection/tree/main/mmdet/evaluation/metrics)
    + [coco_affected_metric.py](coco_affected_metric.py) -> [mmdetection/mmdet/evaluation/metrics](https://github.com/open-mmlab/mmdetection/tree/main/mmdet/evaluation/metrics)
    + [deformable-detr-refine-twostage_r50_16xb2-50e_finebio.py](deformable-detr-refine-twostage_r50_16xb2-50e_finebio.py) -> [configs/deformable_detr](https://github.com/open-mmlab/mmdetection/tree/main/configs/deformable_detr)
    + [dino-4scale_r50_8xb2-12e_finebio.py](dino-4scale_r50_8xb2-12e_finebio.py) -> [configs/dino](https://github.com/open-mmlab/mmdetection/tree/main/configs/dino)


# Pretrained Models
Download [deformable-detr.pth](https://finebio.s3.abci.ai/ckpts/deformable-detr.pth).
 and [dino.pth](https://finebio.s3.abci.ai/ckpts/dino.pth) for quick evaluation.


## Training
#### Deformable DETR
```Shell
python tools/train.py configs/deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_finebio.py
```
#### DINO
```Shell
python tools/train.py configs/dino/dino-4scale_r50_8xb2-12e_finebio.py
```


## Evaluation
#### Deformable DETR
```Shell
python tools/test.py configs/deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_finebio.py [/path/to/model]
```
#### DINO
```Shell
python tools/test.py configs/dino/dino-4scale_r50_8xb2-12e_finebio.py [/path/to/model]
```


Please refer to [the official documentation](https://mmdetection.readthedocs.io/en/latest/get_started.html) if you have further questions.
