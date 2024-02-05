# Manipulated/Affected Object Detection benchmark of the FineBio dataset
This repository provides the code for the Manipulated/Affected Object Detection benchmark of the FineBio dataset. The implemented detecor is based on the paper [Understanding Human Hands in Contact at Internet Scale](https://arxiv.org/pdf/2006.06669.pdf) and this codebase is built on [DINO](https://github.com/IDEA-Research/DINO).

## Requirements
The code has been tested with Python 3.7.16, PyTorch 1.7.1 and CUDA 11.0.
```Shell
conda create --name handobj python=3.7
conda activate handobj
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

cd models/dino/ops
python setup.py build install
# unit test (should see all checking is True)
python test.py
cd ../../..
```

## Data Preparation
+ Download `finebio_coco_annotations.zip` and `finebio_object_detection_images.zip` and unfreeze them in `data` folder under this repository. 
+ `data` folder should be like 
    ```
    data
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

# Pretrained Models
Download [the pretrained object detector](https://finebio.s3.abci.ai/ckpts/dino_checkpoint_e30.pth) and [our pretrained model](https://finebio.s3.abci.ai/ckpts/handobj_checkpoint_e5.pth) for quick evaluation.

## Run
<details>
  <summary>1. Training</summary>

  Run the command below. 
  ```sh
  python main.py --output_dir logs/reproduce/train 
  -c config/DINO/DINO_4scale.py
  --data_path data 
  --frozen_model_path [/path/to/object/detector]
  --use_manipulatedstate 
  --predict_affectedobj 
  --use_affectedstate 
  --options dn_scalar=100 dn_label_coef=1.0 dn_bbox_coef=1.0 epochs=5
  ```

</details>
<details>
  <summary>2. Evaluation</summary>

  Run the command below.
  ```sh
  python main.py --output_dir logs/reproduce/test
  --eval 
  --resume [/path/to/model]
  -c config/DINO/DINO_4scale.py 
  --data_path data 
  --use_manipulatedstate 
  --predict_affectedobj 
  --use_affectedstate 
  --options dn_scalar=100 dn_label_coef=1.0 dn_bbox_coef=1.0
  ```

</details>
<details>
  <summary>3. Demo</summary>

  Run the command below.
  ```sh
  python demo_finebio.py --output_dir demo/det
  -c config/DINO/DINO_4scale.py 
  --image_dir [path/to/image/directory]
  --gt_path data/annotations/v1_test_fpv.json
  --ckpt_path [/path/to/model]
  --use_manipulatedstate 
  --predict_affectedobj 
  --use_affectedstate 
  --options dn_scalar=100 dn_label_coef=1.0 dn_bbox_coef=1.0
  ```

</details>

## Code Details
### Model
Our model is a modified version of [Shan's work](https://github.com/ddshan/hand_object_detector).
We add five branches as follows: 
1. binary hand state classification (manipulating or not)
2. binary manipulated state classification (manipulated or not) 
3. manipulated object offset prediction (offsets from hand location) 
4. binary affecting state classification (affecting or not) 
5. binary affected state classification (afffected or not)  
6. affected object offset prediction (offsets from manipulated object location).  

We implement the model by extending codes of [DINO](https://github.com/IDEA-Research/DINO).
Main model is implemented in [models/dino/dino.py](models/dino/dino.py), and extension modules are in [models/dino/extension_layers.py](models/dino/extension_layers.py).

### Data
Extract required data from annotation files in [datasets/finebio.py](datasets/finebio.py#L324).   
If you don't follow directory structures mentioned [above](#data), you can indicate your annotation path in [datasets/finebio.py](datasets/finebio.py#L694).

### Loss
Loss for training newly-added heads is implemented in [models/dino/dino.py](models/dino/dino.py#L474).

### Inference
How to infer manipulated objects is written in [datasets/finebio_eval.py](datasets/finebio_eval.py#L102). We also decide affected objects in [datasets/finebio_eval.py](datasets/finebio_eval.py#L167).

### Evaluation of Hand AP
We calculate hand ap by four TP criterions as follows:
1. hand location and classification are correct
2. satisfy 1 and hand state is correct
3. satisfy 1, 2 and manipulated object is correct
4. satisfy 1, 2, 3 and affected object is correct  

Corresponding implementation is in [datasets/finebio_eval.py](datasets/finebio_eval.py#L233).



Since this repository is built based on [DINO](https://github.com/IDEA-Research/DINO), please refer to the original repository if you have further questions.
