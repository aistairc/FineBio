# Atomic Operation Detection benchmark of the FineBio dataset

This repository provides the code for the Atomic Operation Detection benchmark of the FineBio dataset: [ActionFormer: Localizing Moments of Actions with Transformers](https://arxiv.org/pdf/2202.07925.pdf).

## Requirements
Follow [INSTALL.md](INSTALL.md) for installing necessary dependencies and compiling the code.

The code has been tested with Python 3.8.0, PyTorch 1.12.1 and CUDA 11.3.  
See [env.yaml](env.yaml) for our environment details.

## Data Preparation
There are two ways to get required input features.
1. Calculate I3D features from scratch
  + Extract I3D features by using [I3D](../../feature_extraction/I3D)  
    **NOTE:** Add `-width 16 -stride 4` in the arguments.
  + Rename the feature folder to `finebio_rgbflow_features` and put it in `data` folder under this repository.
2. Download `finebio_i3d_features_atomic_operation_detection_fpv.zip`
  + Unzip it in `data` folder under this repository.
  + Rename the feature folder to `finebio_rgbflow_features`.

`data` folder will be like this:
```
data/
├── annotations/
    ├── annotation_all.json
    ├── atomic_operation_to_int.json
    ├── verb_to_int.json
    ├── object_to_int.json
    └── fuse_matrix.npy
├── statistics/
    ├── labe2cnt_affected.csv
    ├── labe2cnt_atomic_operation.json
    ├── label2cnt_manipulated.csv
    ├── label2cnt_verb.csv
    ├── labe2duration_affected.csv
    ├── labe2duration_atomic_operation.json
    ├── label2duration_manipulated.csv
    └── label2duration_verb.csv
└── finebio_rgbflow_features/
    ├── P01_01_01.npy
    ├── P01_01_02.npy
    ...
    └── P28_06_01.npy
```

## Run
<details>
  <summary>1. Training</summary>

  #### Single-head model  
  Predict one entity with a single head.
  ```sh
  python train.py configs/finebio_i3d_atomic_operation.yaml --output reproduce
  ```
  + `--apply_graph_modeling`: Specify whether use graph modeling.

  #### Multi-head model  
  Predict multiple entities with multiple heads.
  ```sh
  python train.py configs/finebio_i3d_multi.yaml --output reproduce
  ``` 
  + `--pred_op`: Mean whether additionaly predict operations by combining entity predictions. When you add this, heads should be three each of which is for verb/manipulated/affected.
  + `--op_pred_method`: Indicate how to fuse entity predictions. Choose from 'fuse' or 'cls_head'. 'fuse' multiplies three entity probabilities without adding any additional modules. 'cls_head' uses a classification head which predicts operations by taking concatenated three logits as input.
  + `--apply_graph_modeling`: Specify whether use graph modeling in the additional operation prediction. Unless you add `--pred_op` together, it will be ignored. 

</details>
<details>
  <summary>2. Evaluation</summary>

  #### Single-head model  
  Predict one entity with a single head.  
  ```sh
  python eval_singlehead.py configs/finebio_i3d_atomic_operation.yaml [path/to/ckpt]
  ```
  + `--apply_op_graph_modeling`: Specify whether use graph modeling.

  For atomic operation, you can also evaluate entity results by separating an atomic operation label into verb, manipulated and affected labels.
  ```sh
  python operation_split_eval.py 
  --op_pkl_file [path/to/atomic_operation_eval_results.pkl] 
  --config configs/finebio_i3d_atomic_operation.yaml
  ```

  #### Mult-head model  
  Predict multiple entities with multiple heads.  
  ```sh
  python eval_multihead.py configs/finebio_i3d_multi.yaml [path/to/ckpt] 
  --pred_op --op_pred_method 'fuse'
  ``` 
  + `--pred_op`: Mean whether additionaly predict operations by combining entity predictions. When you add this, heads should be three each of which is for verb/manipulated/affected.
  + `--op_pred_method`: Indicate how to fuse entity predictions. Choose from 'fuse' or 'cls_head'. 'fuse' multiplies three entity probabilities without adding any additional modules. 'cls_head' uses a classification head which predicts operations by taking concatenated three logits as input.
  + `--pivot_type`: We limit the number of detections in inference. If you want to select the detections at the same timepoints for all entities, specify a type whose score should be used. Choose from 'verb' or 'manipulated' or 'affected' or 'atomic_operation'. 
  + `--apply_graph_modeling`: Specify whether use graph modeling in the additional operation prediction. Unless you add `--pred_op` together, it will be ignored.

  #### Combination of single-head models
  Infer operation by combining results from three single-head models for verb, manipulated and affected.
  ```sh
  python combine_severalmodel.py 
  --verb_ckpt [path/to/verb/ckpt]
  --manipulated_ckpt [path/to/manipulated/ckpt]
  --affected_ckpt [path/to/affected/ckpt]
  --output_path [path/to/output/directory]
  ``` 
  + `--pivot_type`:  We limit the number of detections in inference. If you want to select the detections at the same timepoints for all entities, specify a type whose score should be used. Choose from 'verb' or 'manipulated' or 'affected' or 'atomic_operation'. 

</details>

## Code Details
### Model
Single-head and multi-head models are both implemented in [libs/modeling/meta_archs.py](libs/modeling/meta_archs.py). `PtTransformer` and `MultiPredictionPtTransformer` denote single-head and multi-head detector for each.

### Data
Read data in [libs/datasets/aip.py](libs/datasets/aip.py). 

### Evaluation
Evaluation functions are in [libs/utils/train_utils.py](libs/utils/train_utils.py). `valid_one_epoch`, `valid_one_epoch_multi` and `valid_one_epoch_combine` are evaluation functions for a single-head model, a multi-head model, and combination of single-head models, respectively. AP calculation and result visualization are written in [libs/utils/metrics.py](libs/utils/metrics.py).

### FP/FN Analysis
Please refer to [analysis](analysis/DETAD).



Since this repository is built based on [actionformer_release](https://github.com/happyharrycn/actionformer_release), please refer to the original repository if you have further questions.
