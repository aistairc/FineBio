import os
from glob import glob
import argparse

import torch
from torch.autograd import Variable
from torchvision import transforms

import numpy as np
from pytorch_i3d import InceptionI3d
import cv2
from tqdm import tqdm

# image folder inside should be
# P01 (participant id)
# - P01_01_01 (video name)
# -- ~.jpg
# - P01_01_02 
# P02
# - P02_01_01
# video folder inside should be
# P01_01_01.mp4 (fpv)
# P01_01_01_T1.mp4 (tpv)
# P01_01_02.mp4

def load_rgb_frames(frames_dir, max_len, test_transforms=None):
    # setup dataset
    frames = []
    for frame_path in sorted(glob(os.path.join(frames_dir, '*'))):
        frame = cv2.imread(frame_path)[:, :, [2, 1, 0]]
        h, w, c = frame.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            frame = cv2.resize(frame, dsize=(0, 0), fx=sc, fy=sc)
        if max_len != -1 and (w > max_len or h > max_len):
            d = max(w, h) - float(max_len)
            sc = 1 - d / max(w, h)
            frame = cv2.resize(frame, dsize=(0, 0), fx=sc, fy=sc)
        frame = (frame / 255.) * 2 - 1
        frames.append(frame)
    frames = np.asarray(frames, dtype=np.float32)
    if test_transforms:
        frames = test_transforms(frames)
    return torch.from_numpy(frames.transpose([3, 0, 1, 2]))  # T H W C -> C T H W


def load_rgb_video(video_file, max_len, test_transforms=None):
    # setup dataset
    frames = []
    cap = cv2.VideoCapture(video_file)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[:, :, [2, 1, 0]]
        h, w, c = frame.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            frame = cv2.resize(frame, dsize=(0, 0), fx=sc, fy=sc)
        if max_len != -1 and (w > max_len or h > max_len):
            d = max(w, h) - float(max_len)
            sc = 1 - d / max(w, h)
            frame = cv2.resize(frame, dsize=(0, 0), fx=sc, fy=sc)
        frame = (frame / 255.) * 2 - 1
        frames.append(frame)
    frames = np.asarray(frames, dtype=np.float32)
    if test_transforms:
        frames = test_transforms(frames)
    return torch.from_numpy(frames.transpose([3, 0, 1, 2]))  # T H W C -> C T H W


def load_flow_frames(frames_dir, max_len, test_transforms=None):
    frames = []
    for frame_path in sorted(glob(os.path.join(frames_dir, '*_x.png'))):
        imgx = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(frame_path.replace('_x', '_y'), cv2.IMREAD_GRAYSCALE)
        
        h, w = imgx.shape
        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)
        if max_len != -1 and (w > max_len or h > max_len):
            d = max(w, h) - float(max_len)
            sc = 1 - d / max(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)
            
        imgx = (imgx / 255.) * 2 - 1
        imgy = (imgy / 255.) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    frames = np.asarray(frames, dtype=np.float32)
    if test_transforms:
        frames = test_transforms(frames)
    return torch.from_numpy(frames.transpose([3, 0, 1, 2]))  # T H W C -> C T H W


def run(args):
    data_dir = args.data_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    mode = args.mode
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(args.num_classes, in_channels=2)
    elif mode == 'rgb':
        i3d = InceptionI3d(args.num_classes, in_channels=3)
    i3d.load_state_dict(torch.load(args.model))
    i3d.cuda()

    i3d.train(False)  # Set model to evaluate mode
                
    # Iterate over data.
    type = ""
    data_paths = glob(os.path.join(data_dir, "**"), recursive=True)[1:]
    path_lists = []
    for path in data_paths:
        if os.path.isfile(path):
            ext = os.path.splitext(os.path.basename(path))[1]
            if ext in [".png", ".jpg", ".jpeg", ".JPG"]:
                if type == "video":
                    raise AssertionError("Video and image both exist. Format should be consistent.")
                # if file is an image, take parent directory to extract video feature.
                path = os.path.abspath(os.path.join(path, os.pardir))
                type = "image"
            elif ext in [".mp4"]:
                if type == "image":
                    raise AssertionError("Video and image both exist. Format should be consistent.")
                # if view point is not egocentric, view id is specified in the end of video name.
                # video name: {participant id}_{protocol number}_{trial number}_{view id}
                video_name = os.path.splitext(os.path.basename(path))[0]
                view_id = "T0"
                if len(video_name.split('_')) == 4:
                    view_id = video_name.split('_')[3]
                if view_id != args.view_id:
                    continue
                type = "video"
            else:
                raise AssertionError(f"Invalid file format: {ext}.")
            if path in set(path_lists):
                continue
            path_lists.append(path)
    
    path_lists = sorted(path_lists)
    for path in tqdm(path_lists):
        video_name = ""
        if type == "image":
            video_name = os.path.basename(path)
        elif type == "video":
            video_name = os.path.splitext(os.path.basename(path))[0]
            if len(video_name.split('_')) == 4:
                view_id = video_name.split('_')[3]
                video_name = video_name.replace(f"_{view_id}", "")
        par_id = int(video_name.split('_')[0][1:])
        if par_id < args.min_par_id:
            continue
        save_path = os.path.join(save_dir, video_name + '.npy')
        if os.path.exists(save_path):
            continue

        print(path)
        # get the inputs
        if args.mode == "rgb":
            if type == "image":
                frames = load_rgb_frames(path, args.max_len)
            elif type == "video":
                frames = load_rgb_video(path, args.max_len)
        elif args.mode == 'flow':
            if type == "image":
                frames = load_flow_frames(path, args.max_len)
            elif type == "video":
                raise AssertionError("Can't extract flow features from a video.")
        
        c, t, h, w = frames.shape
        features = []
        for frame_idx in range(0, t, args.stride):
            if args.symmetry:
                start = frame_idx - args.width // 2
                end = frame_idx + args.width // 2 + 1
            else:
                start = frame_idx
                end = frame_idx + args.width
            inputs = frames[:, max(0, start):min(end, t), :, :]
            if start < 0:
                dummy = torch.zeros((c, -start, h, w))
                inputs = torch.cat((dummy, inputs), dim=1)
            elif end > t:
                dummy = torch.zeros((c, end - t, h, w))
                inputs = torch.cat((inputs, dummy), dim=1)
            assert inputs.shape[1] == args.width
            inputs = Variable(inputs.unsqueeze(0).cuda(), volatile=True)
            feature = i3d.extract_features(inputs)  # (1, 1024, T/8, H/32, W/32)
            feature = torch.mean(feature.squeeze().view(1024, -1), dim=-1)
            features.append(feature.data.cpu().numpy())
        assert len(features) == (t + args.stride - 1) // args.stride, "Invalid feature length."
        np.save(save_path, np.array(features))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, help='rgb or flow')
    parser.add_argument('-data_dir', type=str)
    parser.add_argument('-model', type=str, default="./models/rgb_imagenet.pt")
    parser.add_argument('-max_len', type=int, help="max length of image size. -1 for full size", default=-1)
    parser.add_argument('-num_classes', type=int, default=400)
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-min_par_id', type=int, help="minimum participant id", default=0)
    parser.add_argument('-view_id', type=str, help="view point id (T0 for egocentric, others for third-person view). Referred only when input format is video.", default='T0')
    parser.add_argument('-save_dir', type=str)
    parser.add_argument('-width', type=int, default=21)
    parser.add_argument('-stride', type=int, default=1)
    parser.add_argument('-symmetry', action="store_true")  # if it's true [t-l/2, t+l/2] else [t, t+l]

    args = parser.parse_args()
    if args.symmetry and args.width % 2 == 0:
        print("Window is symmetry, so width should be odd")
        exit(1)
    if os.path.basename(args.save_dir) != args.view_id:
        # confirm if save_dir matches view id.
        print(f"save_dir is '{args.save_dir}' and input view_id is '{args.view_id}'. Is that OK? If yes push y. Otherwise, push n")
        res = input()
        patience = 5
        while res not in ['y', 'Y', 'n', 'N'] and patience > 0:
            patience -= 1
            print("If yes push y. Otherwise, push n")
            res = input()
        if patience == 0 or res in ['n', 'N']:
            exit(1)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run(args)
