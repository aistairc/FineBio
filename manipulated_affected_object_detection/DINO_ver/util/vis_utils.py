import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from util.utils import renorm
from util.misc import color_sys

from util.viz_hand_obj import *

_color_getter = color_sys(100)

# plot known and unknown box
def add_box_to_img(img, boxes, colorlist, brands=None):
    """[summary]

    Args:
        img ([type]): np.array, H,W,3
        boxes ([type]): list of list(4)
        colorlist: list of colors.
        brands: text.

    Return:
        img: np.array. H,W,3.
    """
    H, W = img.shape[:2]
    for _i, (box, color) in enumerate(zip(boxes, colorlist)):
        x, y, w, h = box[0] * W, box[1] * H, box[2] * W, box[3] * H
        img = cv2.rectangle(img.copy(), (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), color, 2)
        if brands is not None:
            brand = brands[_i]
            org = (int(x-w/2), int(y+h/2))
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            thickness = 1
            img = cv2.putText(img.copy(), str(brand), org, font, 
                fontScale, color, thickness, cv2.LINE_AA)
    return img


def plot_dual_img(img, boxes, labels, idxs, probs=None):
    """[summary]

    Args:
        img ([type]): 3,H,W. tensor.
        boxes (): tensor(Kx4) or list of tensor(1x4).
        labels ([type]): list of ints.
        idxs ([type]): list of ints.
        probs (optional): listof floats.

    Returns:
        img_classcolor: np.array. H,W,3. img with class-wise label.
        img_seqcolor: np.array. H,W,3. img with seq-wise label.
    """

    boxes = [i.cpu().tolist() for i in boxes]
    img = (renorm(img.cpu()).permute(1,2,0).numpy() * 255).astype(np.uint8)
    # plot with class
    class_colors = [_color_getter(i) for i in labels]
    if probs is not None:
        brands = ["{},{:.2f}".format(j,k) for j,k in zip(labels, probs)]
    else:
        brands = labels
    img_classcolor = add_box_to_img(img, boxes, class_colors, brands=brands)
    # plot with seq
    seq_colors = [_color_getter((i * 11) % 100) for i in idxs]
    img_seqcolor = add_box_to_img(img, boxes, seq_colors, brands=idxs)
    return img_classcolor, img_seqcolor


def plot_raw_img(img, boxes, labels):
    """[summary]

    Args:
        img ([type]): 3,H,W. tensor. 
        boxes ([type]): Kx4. tensor
        labels ([type]): K. tensor.

    return:
        img: np.array. H,W,3. img with bbox annos.
    
    """
    img = (renorm(img.cpu()).permute(1,2,0).numpy() * 255).astype(np.uint8)
    H, W = img.shape[:2]
    for box, label in zip(boxes.tolist(), labels.tolist()):
        x, y, w, h = box[0] * W, box[1] * H, box[2] * W, box[3] * H

        img = cv2.rectangle(img.copy(), (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), _color_getter(label), 2)
        # add text
        org = (int(x-w/2), int(y+h/2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 1
        img = cv2.putText(img.copy(), str(label), org, font, 
            fontScale, _color_getter(label), thickness, cv2.LINE_AA)

    return img


def vis_detections_filtered_objects_finebio_PIL(im, dets, thresh_hand=0.8, thresh_obj=0.01, font_path='util/times_b.ttf', predict_affectedobj=False):
    # convert to PIL
    im = im[:,:,::-1]
    image = Image.fromarray(im).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, size=60)
    width, height = image.size 
    
    # class_name, class_box, class_score, handstate, x, y, magnitude, manipulated state, affecting state, affected state
    hand_dets, obj_dets = None, None
    if set(list(dets.keys())) - set(["left_hand", "right_hand"]):
        obj_dets = np.concatenate([np.concatenate([np.array([class_name] * len(dets[class_name]))[:, None], dets[class_name]], axis=1) for class_name in dets if 'hand' not in class_name], axis=0)
        order = np.argsort(obj_dets[:, 5])[::-1]  # sort by score
        obj_dets = obj_dets[order]
    if "left_hand" in dets.keys() or "right_hand" in dets.keys():
        hand_dets = np.concatenate([np.concatenate([np.array([class_name] * len(dets[class_name]))[:, None], dets.get(class_name, [])], axis=1) for class_name in dets if 'hand' in class_name], axis=0)
        order = np.argsort(hand_dets[:, 5])[::-1] # sort by score
        hand_dets = hand_dets[order]
    
    if obj_dets is not None and hand_dets is not None:
        # index of manipulated object in obj_dets
        manip_id = filter_manip_object(obj_dets[:, 1:].astype(np.float32), hand_dets[:, 1:].astype(np.float32))
        # index of affected object in hand_dets/obj_dets. 
        # if < len(hand_dets), the index is for hand_dets. Otherwise, the index - len(hand_dets) is for obj_dets
        affect_id = filter_aff_object(obj_dets[:, 1:].astype(np.float32), hand_dets[:, 1:].astype(np.float32), manip_id)
        for obj_idx, i in enumerate(range(obj_dets.shape[0])):
            bbox = list(int(np.round(float(x))) for x in obj_dets[i, 1:5])
            score = float(obj_dets[i, 5])
            name = obj_dets[i, 0]
            # For objects with higher score than threshold and manipulated by hand.
            if score > thresh_obj and i in manip_id:
                # manipulated object
                name += "-M"  # indicate "Manipulated Obj."
                image = draw_manip_obj_mask(image, draw, obj_idx, bbox, score, width, height, font, txt=name)   
            # For objects with higher score than threshold and affected by manipulated object.
            if predict_affectedobj:
                if score > thresh_obj and (i + hand_dets.shape[0]) in affect_id:
                    # affected object
                    name += ", A"  # indicate "Affected Obj."
                    image = draw_aff_obj_mask(image, draw, obj_idx, bbox, score,  width, height, font, txt=name)
        for hand_idx, i in enumerate(range(np.minimum(10, hand_dets.shape[0]))):
            bbox = list(int(np.round(float(x))) for x in hand_dets[i, 1:5])
            score = float(hand_dets[i, 5])
            state = float(hand_dets[i, 6])
            name = hand_dets[i, 0]
            lr = (name == "right_hand")
            # if state == 1:
            #     name += '*'
            if score > thresh_hand:
                # viz hand by PIL
                image = draw_hand_mask(image, draw, hand_idx, bbox, score, lr, state, width, height, font, txt=name)

                if state > 0 and manip_id[i] != -1: # in contact hand
                    obj_cc, hand_cc =  calculate_center(obj_dets[manip_id[i],1:5].astype(np.float32)), calculate_center(bbox)
                    # viz line by PIL
                    if lr == 0:
                        side_idx = 0
                    elif lr == 1:
                        side_idx = 1
                    draw_line_point_hand2manip(draw, side_idx, (int(hand_cc[0]), int(hand_cc[1])), (int(obj_cc[0]), int(obj_cc[1])))
                    
                    if predict_affectedobj:
                        if affect_id[i] != -1:
                            if affect_id[i] < hand_dets.shape[0]:
                                affect_cc = calculate_center(hand_dets[affect_id[i], 1:5].astype(np.float32))
                                draw_line_point_manip2aff(draw, (int(obj_cc[0]), int(obj_cc[1])), (int(affect_cc[0]), int(affect_cc[1])))
                            else:
                                affect_cc = calculate_center(obj_dets[(affect_id[i] - hand_dets.shape[0]), 1:5].astype(np.float32))
                                draw_line_point_manip2aff(draw, (int(obj_cc[0]), int(obj_cc[1])), (int(affect_cc[0]), int(affect_cc[1])))

    elif hand_dets is not None:
        image = vis_detections_finebio_PIL(im, 'hand', hand_dets, thresh_hand, font_path)
        
    return image


def vis_detections_finebio_PIL(im, class_name, dets, thresh=0.8, font_path='lib/model/utils/times_b.ttf'):
    """Visual debugging of detections."""
    
    image = Image.fromarray(im).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, size=30)
    width, height = image.size
    
    for hand_idx, i in enumerate(range(np.minimum(10, dets.shape[0]))):
        bbox = list(int(np.round(float(x))) for x in dets[i, 1:5])
        score = float(dets[i, 5])
        state = float(dets[i, 6])
        name = dets[i, 0]
        lr = (name == "right_hand")
        # if state == 1:
        #     name += '*'
        if score > thresh:
            image = draw_hand_mask(image, draw, hand_idx, bbox, score, lr, state, width, height, font, txt=name)
            
    return image


def calculate_center(bb):
    return [(bb[0] + bb[2])/2, (bb[1] + bb[3])/2]


def filter_manip_object(obj_dets, hand_dets):
    filtered_object = []
    object_cc_list = []
    object_inds = []
    for j in range(obj_dets.shape[0]):
        if float(obj_dets[j, 9]) == 0:
            # Ignore when state is not manipulated.
            continue
        object_cc_list.append(calculate_center(obj_dets[j,:4]))
        object_inds.append(j)
    object_cc_list = np.array(object_cc_list)
    object_inds = np.array(object_inds)
    
    img_obj_id = [-1 for _ in range(hand_dets.shape[0])]
    if len(object_cc_list):
        for i in range(hand_dets.shape[0]):
            if hand_dets[i, 5] <= 0:  # if hand state is non-manipulating, no manippulated object exists.
                continue
            hand_cc = np.array(calculate_center(hand_dets[i,:4]))
            point_cc = np.array([(hand_cc[0]+hand_dets[i,6]*1000*hand_dets[i,7]), (hand_cc[1]+hand_dets[i,6]*1000*hand_dets[i,8])])
            dist = np.sum((object_cc_list - point_cc)**2,axis=1)
            dist_min = np.argmin(dist)
            img_obj_id[i] = object_inds[dist_min]
    return img_obj_id


def filter_aff_object(obj_dets, hand_dets, manip_obj_id):
    hand_cc_list = []
    object_cc_list = []
    for j in range(hand_dets.shape[0]):
        hand_cc_list.append(calculate_center(hand_dets[j,:4]))
    for j in range(obj_dets.shape[0]):
        object_cc_list.append(calculate_center(obj_dets[j,:4]))
        
    affect_id = [-1 for _ in range(hand_dets.shape[0])]
    for hand_idx in range(hand_dets.shape[0]):
        manip_obj_idx = manip_obj_id[hand_idx]
        if manip_obj_idx == -1:  
            # if there's no manipulated object for the hand, then there's no affected object.
            continue
        if float(obj_dets[manip_obj_idx, 10]) == 0:
            # Ignore when state of the manipulated object is non-affecting.
            continue
        all_cc_list = []
        inds = []
        for j in range(hand_dets.shape[0]):
            if j == hand_idx or hand_dets[j, 11] == 0:
                # if hand is in non-affected state, then it won't be an affeted object.
                continue
            all_cc_list.append(hand_cc_list[j])
            inds.append(j)
        for j in range(obj_dets.shape[0]):
            if j == manip_obj_idx or obj_dets[j, 11] == 0:
                # if the object is in non-affected state, then it won't be an affeted object.
                continue
            all_cc_list.append(object_cc_list[j])
            inds.append(hand_dets.shape[0] + j)
        all_cc_list = np.array(all_cc_list)
        inds = np.array(inds)
        if len(all_cc_list):
            manip_cc = np.array(calculate_center(obj_dets[manip_obj_idx, :4]))
            point_cc = np.array([(manip_cc[0]+obj_dets[manip_obj_idx,6]*1000*obj_dets[manip_obj_idx,7]), (manip_cc[1]+obj_dets[manip_obj_idx,6]*1000*obj_dets[manip_obj_idx,8])])
            dist = np.sum((all_cc_list - point_cc)**2, 1)
            dist_min = np.argmin(dist)
            affect_id[hand_idx] = inds[dist_min]
    return affect_id
