import sys
sys.path.append('core')

import argparse
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import torch

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


DEVICE = 'cuda'


def load_images(image_dir, max_len):
    image_paths = glob(os.path.join(image_dir, '*.png')) + \
        glob(os.path.join(image_dir, '*.jpg'))
    frames = []
    for image_path in sorted(image_paths):
        img = cv2.imread(image_path)
        img = np.array(img[:, :, [2, 1, 0]]).astype(np.uint8)
        h, w, c = img.shape
        if max_len != -1 and (w > max_len or h > max_len):
            d = max(w, h) - float(max_len)
            sc = 1 - d / max(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        frames.append(img[None].to(DEVICE))
        if len(frames) == 2:
            yield frames[0], frames[1]
            yield frames[0], frames[1]
            frames = [frames[1]]


def load_video(video_path, max_len):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        img = np.array(img[:, :, [2, 1, 0]]).astype(np.uint8)
        h, w, c = img.shape
        if max_len != -1 and (w > max_len or h > max_len):
            d = max(w, h) - float(max_len)
            sc = 1 - d / max(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        frames.append(img[None].to(DEVICE))
        if len(frames) == 2:
            yield frames[0], frames[1]
            frames = [frames[1]]
        

def viz(flo, save_file):
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    cv2.imwrite(save_file, flo[:, :, [2,1,0]])
    

def uv(flo, save_x_file, save_y_file):
    UNKNOWN_FLOW_THRESH = 1e9
    flo = flo[0].permute(1,2,0).cpu().numpy()
    u = flo[:, :, 0]
    v = flo[:, :, 1]

    idxUnknown = np.where(np.logical_or(
        abs(u) > UNKNOWN_FLOW_THRESH,
        abs(v) > UNKNOWN_FLOW_THRESH
    ))
    if len(idxUnknown[0]):
        print(idxUnknown)
        print(f"unknown flow is detected in {save_x_file}, {save_y_file}.")
    u[idxUnknown] = 0
    v[idxUnknown] = 0

    rad = np.sqrt(u * u + v * v)
    maxrad = np.max(rad)
    eps = np.finfo(np.float32).eps
    u /= (maxrad + eps)
    v /= (maxrad + eps)
    u = (u * 127.999 + 128).astype('uint8')
    v = (v * 127.999 + 128).astype('uint8')
    cv2.imwrite(save_x_file, u)
    cv2.imwrite(save_y_file, v)


def run(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model_path))

    model = model.module
    model.to(DEVICE)
    model.eval()

    data_dir = args.data_dir
    save_dir = args.save_dir
    
    data_paths = glob(os.path.join(data_dir, "**"), recursive=True)[1:]
    path_lists = []
    for path in data_paths:
        if os.path.isfile(path):
            path_lists.append(path)
    path_lists = sorted(path_lists)
    for path in tqdm(path_lists):
        ext = os.path.splitext(os.path.basename(path))[1]
        save_path = ""
        type = ""
        if ext in [".png", ".jpg"]:
            # if file is an image, take parent directory to extract video feature.
            path = os.path.abspath(os.path.join(path, os.pardir))
            video_name = os.path.basename(path)
            par_id = video_name.split('_')[0]
            if int(par_id[1:]) < args.min_par_id:
                continue
            save_path = os.path.join(save_dir, par_id, video_name)
            type = "image"
        elif ext in [".mp4"]:
            video_name = os.path.splitext(os.path.basename(path))[0]
            view_id = "T0"
            # if view point is not egocentric, view id is specified in the end of video name.
            # video name: {participant id}_{protocol number}_{trial number}_{view id}
            if len(video_name.split('_')) == 4:
                view_id = video_name.split('_')[3]
                video_name = video_name.replace(f"_{view_id}", "")
            if view_id != args.view_id:
                continue
            par_id = video_name.split('_')[0]
            if int(par_id[1:]) < args.min_par_id:
                continue
            save_path = os.path.join(save_dir, par_id, video_name)
            type = "video"
        else:
            raise AssertionError(f"Invalid file format: {ext}.")
        
        if os.path.exists(save_path):
            continue
        print(path)
        with torch.no_grad():
            os.makedirs(save_path, exist_ok=True)
            if type == "image":
                generator = load_images(path, args.max_len)
            elif type == "video":
                generator = load_video(path, args.max_len)
            
            for idx, (frame1, frame2) in enumerate(generator):
                padder = InputPadder(frame1.shape)
                frame1, frame2 = padder.pad(frame1, frame2)

                flow_low, flow_up = model(frame1, frame2, iters=20, test_mode=True)
                
                if args.mode == "RGB":
                    save_file = os.path.join(save_path, f"flow_{str(idx+1).zfill(6)}.png")
                    viz(flow_up, save_file)
                elif args.mode == "UV":
                    save_x_file = os.path.join(save_path, f"flow_{str(idx+1).zfill(6)}_x.png")
                    save_y_file = os.path.join(save_path, f"flow_{str(idx+1).zfill(6)}_y.png")
                    uv(flow_up, save_x_file, save_y_file)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help="restore checkpoint")
    parser.add_argument('--data_dir', help="dataset for evaluation")
    parser.add_argument("--save_dir", help="directory to save", type=str)
    parser.add_argument('--max_len', type=int, help="max length of image size. -1 for full size", default=-1)
    parser.add_argument('--view_id', type=str, help="view point id (T0 for egocentric, others for third-person view). Referred only when input format is video.", default='T0')
    parser.add_argument('--min_par_id', type=int, help="minimum participant id", default=0)
    parser.add_argument("--mode", help="RGB or UV", type=str, default="UV")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    
    run(args)