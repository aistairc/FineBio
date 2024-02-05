import yaml


DEFAULTS = {
    # random seed for reproducibility, a large number is preferred
    "init_rand_seed": 1234567891,
    # dataset loader, specify the dataset here
    "dataset_name": "finebio",
    "devices": ['cuda:0'], # default: single gpu
    "train_split": ('training', ),
    "val_split": ('validation', ),
    "model_name": "LocPointTransformer",
    "dataset": {
        # temporal stride of the feats
        "feat_stride": 16,
        # number of frames for each feat
        "num_frames": 32,
        # default fps, may vary across datasets; Set to none for read from json file
        "default_fps": None,
        # input feat dim
        "input_dim": 2304,
        # downsampling rate of features, 1 to use original resolution
        "downsample_rate": 1,
        # max sequence length during training
        "max_seq_len": 2304,
        # threshold for truncating an action
        "trunc_thresh": 0.5,
        # set to a tuple (e.g., (0.9, 1.0)) to enable random feature cropping
        # might not be implemented by the dataloader
        "crop_ratio": None,
        # list of types used in annotation
        # only used for finebio
        "type_names": ["verb"],
        # hand annotation ([None], [left], [right], [left, right])
        # only used for finebio
        "hands": [None]
    },
    "loader": {
        "batch_size": 8,
        "num_workers": 4,
    },
    # network architecture
    "model": {
        # type of backbone (convTransformer | conv)
        "backbone_type": 'convTransformer',
        # type of FPN (fpn | identity)
        "fpn_type": "identity",
        "backbone_arch": (2, 2, 5),
        # scale factor between pyramid levels
        "scale_factor": 2,
        # regression range for pyramid levels
        "regression_range": [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        # number of heads in self-attention
        "n_head": 4,
        # window size for self attention; <=1 to use full seq (ie global attention)
        "n_mha_win_size": -1,
        # kernel size for embedding network
        "embd_kernel_size": 3,
        # (output) feature dim for embedding network
        "embd_dim": 512,
        # if attach group norm to embedding network
        "embd_with_ln": True,
        # feat dim for FPN
        "fpn_dim": 512,
        # if add ln at the end of fpn outputs
        "fpn_with_ln": True,
        # starting level for fpn
        "fpn_start_level": 0,
        # feat dim for head
        "head_dim": 512,
        # kernel size for reg/cls/center heads
        "head_kernel_size": 3,
        # number of layers in the head (including the final one)
        "head_num_layers": 3,
        # if attach group norm to heads
        "head_with_ln": True,
        # defines the max length of the buffered points
        "max_buffer_len_factor": 6.0,
        # disable abs position encoding (added to input embedding)
        "use_abs_pe": False,
        # use rel position encoding (added to self-attention)
        "use_rel_pe": False,
    },
    "train_cfg": {
        # radius | none (if to use center sampling)
        "center_sample": "radius",
        "center_sample_radius": 1.5,
        "loss_weight": 1.0, # on reg_loss, use -1 to enable auto balancing
        "cls_prior_prob": 0.01,
        "init_loss_norm": 2000,
        # gradient cliping, not needed for pre-LN transformer
        "clip_grad_l2norm": -1,
        # cls head without data (a fix to epic-kitchens / thumos)
        "head_empty_cls": [],
        # dropout ratios for tranformers
        "dropout": 0.0,
        # ratio for drop path
        "droppath": 0.1,
        # if to use label smoothing (>0.0)
        "label_smoothing": 0.0,
    },
    "test_cfg": {
        "pre_nms_thresh": 0.001,
        "pre_nms_topk": 5000,
        "iou_threshold": 0.1,
        "min_score": 0.01,
        "max_seg_num": 1000,
        "nms_method": 'soft', # soft | hard | none
        "nms_sigma" : 0.5,
        "duration_thresh": 0.05,
        "multiclass_nms": True,
        "ext_score_file": None,
        "voting_thresh": 0.75,
    },
    # optimizer (for training)
    "opt": {
        # solver
        "type": "AdamW", # SGD or AdamW
        # solver params
        "momentum": 0.9,
        "weight_decay": 0.0,
        "learning_rate": 1e-3,
        # excluding the warmup epochs
        "epochs": 30,
        # lr scheduler: cosine / multistep
        "warmup": True,
        "warmup_epochs": 5,
        "schedule_type": "cosine",
        # in #epochs excluding warmup
        "schedule_steps": [],
        "schedule_gamma": 0.1,
    }
}

def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v

def load_default_config():
    config = DEFAULTS
    return config

def _update_config(config):
    # convert type of dataset/type_names into list
    if type(config["dataset"]["type_names"]) == str:
        config["dataset"]["type_names"] = [config["dataset"]["type_names"]]
    # fill in derived fields
    config["model"]["input_dim"] = config["dataset"]["input_dim"]
    config["model"]["max_seq_len"] = config["dataset"]["max_seq_len"]
    config["model"]["train_cfg"] = config["train_cfg"]
    config["model"]["test_cfg"] = config["test_cfg"]
    return config

def check_config(config):
    # check if some items in config meet regulations
    # only used for finebio
    valid = True
    message = ""

    hands = []
    for hand in config['dataset']['hands']:
        if hand not in ["left", "right", None]:
            message += f"config['dataset']['hands']={hand} should not include anything except: ['left', 'right']\n"
            valid = False
            break
        # remove duplicate types
        if hand not in hands:
            hands.append(hand)
    config['dataset']['hands'] = hands
    
    # in single model, left and right annotation can't be predicted separately
    if set(["left", "right"]) <= set(config['dataset']['hands']) and config['model_name'] == "LocPointTransformer":
        message += "LocPointTransformer can't predict left and right hand separately. Set either [left] or [right] or []" 
        valid = False

    type_names = []
    for type_name in config['dataset']['type_names']:
        if type_name not in ["verb", "manipulated", "affected", "atomic_operation", "hand"]:
            message += f"config['dataset']['type_names'] should be subset of ['verb', 'manipulated', 'affected', 'atomic_operation', 'hand]. {type_name} is invalid.\n"
            valid = False
            break
        # remove duplicate types
        if type_name not in set(type_names):
            type_names.append(type_name)
    config['dataset']['type_names'] = type_names
    
    # if len(type_names) > 1 and config['model_name'] == "LocPointTransformer":
    #     message += "LocPointTransformer has only one classification head, so config['dataset']['type_names'] should be single."
    #     valid = False
    if len(type_names) == 1 and config['model_name'] == "MultiPredictionPtTransformer":
        message += "MultiPredictionPtTransformer has multiple classification heads, so config['dataset']['type_names'] should be multiple."
        valid = False

    if not valid:
        raise AssertionError(message)


def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    _merge(defaults, config)
    config = _update_config(config)
    check_config(config)
    return config