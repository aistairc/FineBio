# The new config inherits a base config to highlight the necessary modification
_base_ = '../deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    bbox_head=dict(num_classes=35))

# save folder
work_dir = './outputs/deformable_detr'
# Modify dataset related settings
# TODO: put the data folder path below.
data_root = './'
metainfo = {
    'classes': (
        "left_hand",
        "right_hand",
        "blue_pipette",
        "yellow_pipette",
        "red_pipette",
        "8_channel_pipette",
        "blue_tip",
        "yellow_tip",
        "red_tip",
        "8_channel_tip",
        "blue_tip_rack", 
        "yellow_tip_rack",
        "red_tip_rack",
        "8_channel_tip_rack",
        "50ml_tube",
        "15ml_tube",
        "micro_tube",
        "8_tube_stripes",
        "8_tube_stripes_lid",
        "50ml_tube_rack",
        "15ml_tube_rack",
        "micro_tube_rack",
        "8_tube_stripes_rack",
        "8_tube_stripes_rack_lid",
        "cell_culture_plate",
        "cell_culture_plate_lid",
        "trash_can",
        "centrifuge",
        "vortex_mixer",
        "magnetic_rack",
        "pcr_machine",
        "tube_with_spin_column",
        "spin_column",
        "tube_without_lid",
        "pen"
        ),
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=f'annotations/v1_train_fpv.json',
        data_prefix=dict(img='images/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=f'annotations/v1_valid_fpv.json',
        data_prefix=dict(img='images/')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=f'annotations/v1_test_fpv.json',
        data_prefix=dict(img='images/')))

# Modify metric related settings
val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + f'annotations/v1_valid_fpv.json',
        metric='bbox',
        format_only=False,
        backend_args=None
    ),
    # dict(
    #     type='CocoManipulatedMetric',
    #     ann_file=data_root + f'annotations/v1_valid_fpv.json',
    #     metric='bbox',
    #     format_only=False,
    #     backend_args=None
    # ),
    # dict(
    #     type='CocoAffectedMetric',
    #     ann_file=data_root + f'annotations/v1_valid_fpv.json',
    #     metric='bbox',
    #     format_only=False,
    #     backend_args=None
    # ),
]
test_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + f'annotations/v1_test_fpv.json',
        metric='bbox',
        format_only=False,
        backend_args=None,
        classwise=True
    ),
    dict(
        type='CocoManipulatedMetric',
        ann_file=data_root + f'annotations/v1_test_fpv.json',
        metric='bbox',
        format_only=False,
        backend_args=None,
        classwise=True
    ),
    dict(
        type='CocoAffectedMetric',
        ann_file=data_root + f'annotations/v1_test_fpv.json',
        metric='bbox',
        format_only=False,
        backend_args=None,
        classwise=True
    ),
]

# We can use the pre-trained deformable DETR model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_coco/deformable-detr-refine-twostage_r50_16xb2-50e_coco_20221021_184714-acc8a5ff.pth'
