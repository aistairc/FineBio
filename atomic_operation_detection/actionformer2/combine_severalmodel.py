# python imports
import argparse
import os
from glob import glob
import time
from pprint import pprint
import numpy as np

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch, PredictionFuser
from libs.utils import valid_one_epoch_combine, ANETdetection, fix_random_seed


def main(args):
    # user settings
    cfgs = [args.verb_config, args.manipulated_config, args.affected_config]
    ckpts = [args.verb_ckpt, args.manipulated_ckpt, args.affected_ckpt]
    type_names = ["verb", "manipulated", "affected"]
    hand = args.hand
    #############################################################################################
    assert (args.pivot_type is None) or (args.pivot_type in ["verb", "manipulated", "affected", "atomic_operation"]), "score_by should be one of ['verb', 'manipulated', 'affected', 'atomic_operation']"
    """0. load config"""
    # sanity check
    cfg_dict = {}
    for type_name, config_path in zip(type_names, cfgs):
        if os.path.isfile(config_path):
            cfg = load_config(config_path)
        else:
            raise ValueError("Config file does not exist.")
        assert len(cfg['val_split']) > 0, "Test set must be specified!"
        cfg_dict[type_name] = cfg
    ckpt_dict = {}
    for type_name, ckpt_path in zip(type_names, ckpts):
        assert os.path.isfile(ckpt_path), f"CKPT file does not exist!: {ckpt_path}"
        ckpt_dict[type_name] = ckpt_path

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. add atomic operation config"""
    # add atomic operation
    type_names.append("atomic_operation")
    # randomly copy config for atomic operation
    cfg_dict[type_names[-1]] = cfg_dict[type_names[0]].copy()
        
    """3. create dataset / dataloader"""
    dataset_dict = {}
    dataloaders = []  # for verb/manipulated/affected/atomic_operation
    num_classes= []  # for verb/manipulated/affected/atomic_operation
    for type_name in type_names:
        cfg_dict[type_name]['dataset_name'] == "finebio"
        cfg_dict[type_name]["dataset"]["type_names"] = [type_name]
        cfg_dict[type_name]["dataset"]["hands"] = [hand] if hand != '' else []
        val_dataset = make_dataset(
            cfg_dict[type_name]['dataset_name'], False, cfg_dict[type_name]['val_split'], **cfg_dict[type_name]['dataset']
        )
        dataset_dict[type_name] = val_dataset
        dataloaders.append(
            make_data_loader(
                val_dataset, False, None, 1, cfg_dict[type_name]['loader']['num_workers']
            )
        )
        num_classes.append(val_dataset.num_classes[0])

    """4. create model and evaluator"""
    models = []  # verb/manipulated/affected
    for type_name in type_names:
        # no model for atomic operation
        if "atomic_operation" in type_name:
            continue
        # model (no multi GPU training)
        # model should be 'LocPointTransformer'
        assert cfg_dict[type_name]['model_name'] == "LocPointTransformer", "No model can't be handled except LocPointTransformer"
        cfg_dict[type_name]["model"]["num_classes"] = dataset_dict[type_name].num_classes[0]
        model = make_meta_arch(cfg_dict[type_name]['model_name'], **cfg_dict[type_name]['model'])
        model = nn.DataParallel(model, device_ids=cfg_dict[type_name]['devices'])
        # load ckpt
        print("=> loading {} checkpoint '{}'".format(type_name, ckpt_dict[type_name]))
        # load ckpt, reset epoch / best rmse
        checkpoint = torch.load(
            ckpt_dict[type_name],
            map_location=(lambda storage, loc: storage.cuda(cfg_dict[type_name]['devices'][0]))
        )
        # load ema model instead
        print("Loading from EMA model ...")
        model.load_state_dict(checkpoint['state_dict_ema'])
        models.append(model)
        del checkpoint

    # set up evaluator
    os.makedirs(args.output_path, exist_ok=True)
    det_evals, output_files = [], []
    for type_name in type_names:
        val_db_vars = dataset_dict[type_name].get_attributes()
        if not args.saveonly:
            det_evals.append(ANETdetection(
                dataset_dict[type_name].data_lists[0],
                dataset_dict[type_name].split[0],
                tiou_thresholds=val_db_vars['tiou_thresholds'],
                dataset_name=type_name,
                label_dict=dataset_dict[type_name].label_dicts[0],
                stat_dir=args.stat_dir
            ))
        output_files.append(os.path.join(args.output_path, f'{type_name}_eval_results.pkl'))

    """5. Test the model"""
    print("\nStart testing model LocPointTransformer ...")
    start = time.time()
    
    # fusion parameters
    fuse_mat = torch.from_numpy(np.load("data/annotations/fuse_matrix.npy")).to(torch.float32)
    fuse_weights = [1.0/3, 1.0/3, 1.0/3]
    fuse_mat[fuse_mat.to(torch.bool)] = torch.tensor(fuse_weights).repeat(fuse_mat.shape[0])
    fuse_weight_mats = [fuse_mat.cuda()]
    fuse_groups = [[0, 1, 2]]
    # pivot type (str->idx) if any
    pivot_type = None
    for i, type_name in enumerate(type_names):
        if args.pivot_type == type_name:
            pivot_type = i
    
    assert len(type_names) == len(det_evals) == len(output_files)
    assert len(fuse_groups) == len(fuse_weight_mats) == 1
    
    print(f"test cfg for {type_names[0]} is used.")
    valid_one_epoch_combine(
        cfg_dict[type_names[0]],
        dataloaders,
        models,
        -1,
        type_names,
        num_classes,
        PredictionFuser(fuse_groups, fuse_weight_mats),
        pivot_type=pivot_type,
        evaluator=det_evals,
        output_files=output_files,
        print_freq=args.print_freq
    )
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument("--verb_ckpt", type=str, help="path for verb model ckpt", required=True)
    parser.add_argument("--manipulated_ckpt", type=str, help="path for manipulated object model ckpt", required=True)
    parser.add_argument("--affected_ckpt", type=str, help="path for affected object model ckpt", required=True)
    parser.add_argument('--stat_dir', type=str, metavar='DIR',
                        default="data/statistics",
                        help='path to a directory for statistics')
    parser.add_argument("--hand", type=str, choices=["left", "right", ""], default="")
    parser.add_argument("--verb_config", type=str, help="path for verb model config", default="configs/finebio_i3d_verb.yaml")
    parser.add_argument("--manipulated_config", type=str, help="path for manipulated object model config", default="configs/finebio_i3d_manipulated.yaml")
    parser.add_argument("--affected_config", type=str, help="path for affected object model config", default="configs/finebio_i3d_affected.yaml")
    parser.add_argument("--pivot_type", type=str, help="type name which NMS is done by.")
    parser.add_argument("--output_path", type=str, default='.', help='output path')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    args = parser.parse_args()
    main(args)
