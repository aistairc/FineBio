# Step Segementation benchmark of the FineBio dataset using ASFormer

This repository provides the code for the Step Segmentation benchmark of the FineBio dataset using [ASFormer: Transformer for Action Segmentation](https://arxiv.org/pdf/2110.08568.pdf) .


## Requirements
The code has been tested with Python 3.10.9, PyTorch 1.12.1 and CUDA 11.3.  
See [env.yaml](env.yaml) for our environment details.


## Data Preparation
There are two ways to get required input features.
1. Calculate I3D features from scratch
  + Extract I3D features by using [I3D](../../feature_extraction/I3D)  
    **NOTE:** Add `-width 21 -stride 1 -symmetry` in the arguments.
  + Rename the feature folder to `rgbflow_i3d_features` and put it in `data` folder under this repository.
2. Download `finebio_i3d_features_step_segmentation_fpv.zip` and unfreeze it in `data` folder under this repository.


## Pretrained Model
Download the pretrained model from [here](https://finebio.s3.abci.ai/ckpts/asformer.model).


## Training
Run
```Shell
python train.py \
--run_dir .. 
```

## Evaluation
First inference on the test split by running
```Shell
python inference.py \
--run_dir .. \
--checkpoint [/path/to/model] \
--test_subset test
```
Then evaluate by running
```Shell
python eval.py \
--run_dir .. \
--result_dir ./results/test/all_5d_layer11 \
--test_subset test
```
You can see the results on the validation set by running the two commands with `--test_subset valid`.


Since this repository is built based on [ASFormer](https://github.com/ChinaYi/ASFormer), please refer to the original repository if you have further questions.