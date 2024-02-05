# Step Segementation benchmark of the FineBio dataset using MS-TCN++

This repository provides the code for the Step Segmentation benchmark of the FineBio dataset using [MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation](https://arxiv.org/pdf/2006.09220.pdf).


## Requirements
The code has been tested with Python 3.10.9, PyTorch 1.12.1 and CUDA 11.3.  
See [env.yaml](env.yaml) for our environment details.


## Data Preparation
There are two ways to get required input features.
1. Calculate I3D features from scratch
  + Extract I3D features by using [I3D](../../feature_extraction/I3D)  
    **NOTE:** Add `-width 21 -stride 1 -symmetry` in the arguments.
  + Rename the feature folder to `rgbflow_i3d_features` and put it in `data` folder under this repository.
2. Download `finebio_i3d_features_step_segmentation_fpv.zip` and unzip it at the `data` folder under this repository.


## Pretrained Model
The pretrained model can be found [here](https://finebio.s3.abci.ai/ckpts/mstcn.model).

## Training
Run
```Shell
python train.py \
--run_dir .. \
--num_layers_PG 12 \
--num_layers_R 13 \
--num_R 4 \
--num_epochs 100 \
--lr 0.0005
```

## Evaluation
First inference on the test split by running
```Shell
python inference.py \
--run_dir .. \
--checkpoint [/path/to/model] \
--num_layers_PG 12 \
--num_layers_R 13 \
--num_R 4 \
--lr 0.0005 \
--test_subset test
```
Then evaluate by running
```Shell
python eval.py \
--run_dir .. \
--result_dir ./results/test/all_lr5e-04_PG12_R13*4 \
--test_subset test
```
You can see the results on the validation set by running the two commands with `--test_subset valid`.

Since this repository is built based on [MS-TCN2](https://github.com/sj-li/MS-TCN2), please refer to the original repository it if you have further questions.