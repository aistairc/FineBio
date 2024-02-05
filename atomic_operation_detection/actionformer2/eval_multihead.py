# python imports
import argparse
import os
import glob
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
from libs.utils import valid_one_epoch_multi, ANETdetection, fix_random_seed


################################################################################
def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)
    
    assert (args.pivot_type is None) or (args.pivot_type in ["verb", "manipulated", "affected", "atomic_operation"]), "score_by should be one of ['verb', 'manipulated', 'affected', 'atomic_operation']"

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # num_classes for all types to predict in the model.
    cfg["model"]["num_classes"] = val_dataset.num_classes.copy()
    
    # when operation is also inferenced from verb, manipulated and affected.
    if args.pred_op:
        assert len(set(val_dataset.original_type_names) - set(["verb", "manipulated", "affected"])) == 0, \
            "Model should have three heads (verb/manipulated/affected) to predict operations based on entities."
        assert len(val_dataset.hands) == 1, "all types should correspond to the same side of hand."
        dataset_config = cfg["dataset"].copy()
        dataset_config["type_names"] = ["atomic_operation"]
        action_dataset = make_dataset(
            cfg['dataset_name'], False, cfg['val_split'], **dataset_config
        )
    if not args.pred_op and args.apply_graph_modeling:
        print("WARNING: Operation graph modeling is enabled only when operation is predicted. Please add an argument '--pred_op'.")
    cfg["model"]["with_op_pred"] = args.pred_op
    cfg["model"]["op_pred_method"] = args.op_pred_method
    cfg["model"]["apply_op_graph_modeling"] = args.apply_graph_modeling

    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    pprint(cfg)

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    # set up evaluator
    det_evals, output_files = [], []
    val_db_vars = val_dataset.get_attributes()
    os.makedirs(os.path.join(os.path.split(ckpt_file)[-2], 'results'), exist_ok=True)
    if not args.saveonly:
        det_evals = []
        for i, type_name in enumerate(val_dataset.type_names):
            det_evals.append(ANETdetection(
                val_dataset.data_lists[i],
                val_dataset.split[0],
                tiou_thresholds=val_db_vars['tiou_thresholds'],
                dataset_name=type_name,
                label_dict=val_dataset.label_dicts[i],
                stat_dir=args.stat_dir
            ))
        if args.pred_op:
            for i, type_name in enumerate(action_dataset.type_names):
                det_evals.append(ANETdetection(
                    action_dataset.data_lists[i],
                    action_dataset.split[0],
                    tiou_thresholds=val_db_vars['tiou_thresholds'],
                    dataset_name=type_name,
                    label_dict=action_dataset.label_dicts[i],
                    stat_dir=args.stat_dir
                ))
    output_files = [os.path.join(os.path.split(ckpt_file)[-2], 'results', f'{type_name}_eval_results.pkl') for type_name in val_dataset.type_names]
    if args.pred_op:
        output_files += [os.path.join(os.path.split(ckpt_file)[-2], 'results', f'{type_name}_eval_results.pkl') for type_name in action_dataset.type_names]

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    
    type_names = val_dataset.type_names.copy()
    original_type_names = val_dataset.type_names.copy()
    num_classes = val_dataset.num_classes.copy()
    type_groups_with_same_hand = val_dataset.type_groups_with_same_hand.copy()
    if args.pred_op:
        # update type_names/original_type_names, num_classes
        type_names = type_names + action_dataset.type_names
        original_type_names = original_type_names + action_dataset.original_type_names
        num_classes = num_classes + action_dataset.num_classes
        # add action index in type_groups_with_same_hand
        type_groups_with_same_hand[0] = type_groups_with_same_hand[0] + [3]
            
    # convert format of pivot_type from str to index
    pivot_types = []
    if args.pivot_type:
        for i, original_type_name in enumerate(original_type_names):
            if args.pivot_type == original_type_name:
                pivot_types.append(i)
        assert len(pivot_types) == len(type_groups_with_same_hand)
        print(f"Pivot type: {[type_names[i] for i in pivot_types]}")
        
    print(type_names)
    valid_one_epoch_multi(
        cfg,
        val_loader,
        model,
        -1,
        type_names,
        num_classes,
        type_groups_with_same_hand,
        pivot_types=pivot_types,
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
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('--stat_dir', type=str, metavar='DIR',
                        default="data/statistics",
                        help='path to a directory for statistics')
    parser.add_argument("--pred_op", action="store_true", help="whether predict operation by fusing entity predictions")
    parser.add_argument("--op_pred_method", type=str, choices=["fuse", "cls_head"], default="fuse", help="how to fuse entity predictins into operation prediction")
    parser.add_argument("--apply_graph_modeling", action="store_true", help="whether apply operation graph modeling")
    parser.add_argument("--pivot_type", type=str, choices=["verb", "manipulated", "affected", "atomic_operation"], help="type name which NMS is done by.")
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    args = parser.parse_args()
    main(args)
