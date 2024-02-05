import pickle
import json
import os
import numpy as np
import argparse


def convert_pkl2json(action_pred_pkl_file, atomop2int_file, output_path):
    op2int = json.load(open(atomop2int_file, 'r'))
    int2op = dict(zip(op2int.values(), op2int.keys()))
    with open(action_pred_pkl_file, 'rb') as f:
        data = pickle.load(f)
        results = {}
        for i, vid in enumerate(data['video-id']):
            if vid not in results:
                results[vid] = []
            results[vid].append({
                "score": float(data['score'][i]),
                "segment": [float(data['t-start'][i]), float(data['t-end'][i])],
                "label": int2op[data['label'][i]]
            })
    
    output_path = output_path if output_path else './atomic_operation_eval_results.json'
    with open(output_path, 'w') as f:
        json.dump({"results": results}, f)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, help="path for a pickle file outputted from ActionFormer")
    parser.add_argument("--label_to_int", type=str, help="path for a json file of label dictionary")
    parser.add_argument("--output_dir", type=str, help="path for output directory")
    args = parser.parse_args()
    convert_pkl2json(args.pkl_path, args.label_to_int, os.path.join(args.output_dir, os.path.basename(args.pkl_path).replace(".pkl", ".json")))
    