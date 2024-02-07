# FineBio: A Fine-Grained Video Dataset of Biological Experiments with Hierarchical Annotation

This repository includes the download instuction and the code for the paper [FineBio: A Fine-Grained Video Dataset of Biological Experiments with Hierarchical Annotation](https://arxiv.org/abs/2402.00293). 

![teaser_v2](https://github.com/aistairc/FineBio/assets/6857543/642def1e-34b5-46df-bb33-22d7ae2c7c56)

## Download
Usage is limited to non-commercial research/development use.
Users need to sign the license form to access the videos, metadata and annotation. 
Please sign to the [license agreement](https://finebio.s3.abci.ai/FineBio_License_Agreement.pdf) and submit via [this form](https://forms.gle/Ts64vrG6n2i1fEWCA).

Link to the dataset and required credentials will be sent by e-mail after approval.

## Dataset

### Name convention
Video and annotation files are named by the following rules:
```
# First-person view
P<participant_id (1-32)>_<protocol_id (1-7)>_<take_id>.mp4
# Third-person view
P<participant_id (1-32)>_<protocol_id (1-7)>_<take_id>_T<camera_id (1-5)>.mp4
# Atomic operation annotation
P<participant_id (1-32)>_<protocol_id (1-7)>_<take_id>.txt
# Object detection images
P<participant_id (1-32)>_<protocol_id (1-7)>_<take_id>_<frame_num>.jpg
P<participant_id (1-32)>_<protocol_id (1-7)>_<take_id>__T<camera_id (1-5)>_frame_num>.jpg
```

## Pre-trained Models

Please refer to the README for each benchmark for details.

### Step segmentation
| Backbone  | Model |  Acc  | Edit | F1@10 | F1@25 | F1@50 | F1@75 | Weights | 
| ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |
| I3D | MS-TCN++ |90.2|96.7|97.4|96.7|93.5|73.4| [Link](https://finebio.s3.abci.ai/ckpts/mstcn.model) |
| I3D | ASFormer |87.2|94.8|94.2|92.7|86.5|67.0| [Link](https://finebio.s3.abci.ai/ckpts/asformer.model) |

### Atomic Operation Detection
| Backbone  | Model | mAP@0.3 |0.4|0.5|0.6|0.7|Avg.| Weights | 
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| I3D | ActionFormer　(multi-head + set classification) |45.2|41.7|36.5|28.4|18.7|34.1| [Link](https://finebio.s3.abci.ai/ckpts/actionformer.pth.tar) |

### Object Detection
| Model | AP | AP50 | AP\_manipulated | AP\_affected | Weights |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| DINO | 53.3 | 77.4 | 55.9 | 51.6 | [Link](https://finebio.s3.abci.ai/ckpts/dino.pth) |
| Deformable DETR | 56.1 | 78.5 | 64.0 | 58.8 | [Link](https://finebio.s3.abci.ai/ckpts/deformable-detr.pth) |

### Manipulated/Affected Object Detection
|    | Hand | H + Manipulated | H + M + Affected | Weights |
| ------------- | ------------- | ------------- | -------------| -------------|
| Left Hand | 96.8 | 6.5 | 5.9 | [Link](https://finebio.s3.abci.ai/ckpts/handobj_checkpoint_e5.pth) |
| Right Hand | 94.5 | 22.2 | 10.7 | |

### Previous methods used in the implementation

Please cite the necessary methods if you used our baseline models.

* [I3D](https://arxiv.org/abs/1705.07750.pdf)
* [RAFT](https://arxiv.org/pdf/2003.12039.pdf)
* [MS-TCN++](https://arxiv.org/pdf/2006.09220.pdf)
* [ASFormer](https://arxiv.org/pdf/2110.08568.pdf)
* [ActionFormer](https://arxiv.org/pdf/2202.07925.pdf)
* [Deformable DETR](https://arxiv.org/pdf/2010.04159.pdf)
* [DINO](https://arxiv.org/pdf/2203.03605.pdf)
* [Hand Object Detector](https://arxiv.org/pdf/2006.06669.pdf)

## Support
If you find any problem, please report to Takuma Yagi (takuma.yagi[at]aist.go.jp) or by this repository's issue.

## Citation
Please cite our work if you have used our data or code:
```
@article{yagi2024finebio,
  title={Finebio: a fine-grained video dataset of biological experiments with hierarchical annotation},
  author={Yagi, Takuma and Ohashi, Misaki and Huang, Yifei and Furuta, Ryosuke and Adachi, Shungo and Mitsuyama, Toutai and Sato, Yoichi},
  year={2024},
  journal={arXiv preprint arXiv:2402.00293},
}
```

## License
The code is licensed under the MIT License.
