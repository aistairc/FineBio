import os, json, glob, random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
random.seed(0)

def ratio2coord(ratio, width, height): 
    """
    ratio = [x1, y1, x2, y2]
    return image infos
    """

    x1, y1, x2, y2 = int(float(ratio[0])*width), int(float(ratio[1])*height), int(float(ratio[2])*width), int(float(ratio[3])*height)

    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, width)
    y2 = min(y2, height)
    
    bbox = [x1, y1, x2, y2]

    return bbox

def bbox2center(bbox):
    return (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2))


def draw_manip_obj_mask(image, draw, obj_idx, obj_bbox, obj_score, width, height, font, txt='O'):

    mask = Image.new('RGBA', (width, height))
    pmask = ImageDraw.Draw(mask)
    pmask.rectangle(obj_bbox, outline=manip_rgb, width=10, fill=manip_rgba) 
    image.paste(mask, (0,0), mask)  

    draw.rectangle([min(obj_bbox[0], width - (30*len(txt))), max(0, obj_bbox[1]-60), min(obj_bbox[0], width - (30*len(txt))) + 30*len(txt), max(0, obj_bbox[1]-60)+60], fill=(255, 255, 255), outline=manip_rgb, width=4)
    draw.text((min(obj_bbox[0], width - (30*len(txt)))+5, max(0, obj_bbox[1]-60)-2), txt, font=font, fill=(0,0,0)) #

    return image

def draw_aff_obj_mask(image, draw, obj_idx, obj_bbox, obj_score, width, height, font, txt='O'):

    mask = Image.new('RGBA', (width, height))
    pmask = ImageDraw.Draw(mask)
    pmask.rectangle(obj_bbox, outline=aff_rgb, width=10, fill=aff_rgba) 
    image.paste(mask, (0,0), mask)  

    draw.rectangle([min(obj_bbox[0], width - (30*len(txt))), max(0, obj_bbox[1]-60), min(obj_bbox[0], width - (30*len(txt))) + 30*len(txt), max(0, obj_bbox[1]-60)+60], fill=(255, 255, 255), outline=aff_rgb, width=4)
    draw.text((min(obj_bbox[0], width - (30*len(txt)))+5, max(0, obj_bbox[1]-60)-2), txt, font=font, fill=(0,0,0)) #

    return image


def draw_hand_mask(image, draw, hand_idx, hand_bbox, hand_score, side, state, width, height, font, txt=None):

    if side == 0:
        side_idx = 0
    elif side == 1:
        side_idx = 1
    mask = Image.new('RGBA', (width, height))
    pmask = ImageDraw.Draw(mask)
    pmask.rectangle(hand_bbox, outline=hand_rgb[side_idx], width=10, fill=hand_rgba[side_idx])
    image.paste(mask, (0,0), mask)
    if txt is None:
        txt = f'{side_map3[int(float(side))]}-{state_map2[int(float(state))]}'
    # text
    
    draw = ImageDraw.Draw(image)
    draw.rectangle([min(hand_bbox[0], width - (30*len(txt))), max(0, hand_bbox[1]-60), min(hand_bbox[0], width - (30*len(txt))) + 290, max(0, hand_bbox[1]-60)+60], fill=(255, 255, 255), outline=hand_rgb[side_idx], width=4)
    draw.text((min(hand_bbox[0], width - (30*len(txt)))+6, max(0, hand_bbox[1]-60)-2), txt, font=font, fill=(0,0,0)) # 

    return image
    
def draw_line_point_hand2manip(draw, side_idx, hand_center, object_center):
    # hand > object
    draw.line([hand_center, object_center], fill=manip_rgb, width=10)
    x, y = hand_center[0], hand_center[1]
    r=10
    draw.ellipse((x-r, y-r, x+r, y+r), fill=hand_rgb[side_idx])
    x, y = object_center[0], object_center[1]
    draw.ellipse((x-r, y-r, x+r, y+r), fill=manip_rgb)

def draw_line_point_manip2aff(draw, object_center, object_center2):
    # object > object
    draw.line([object_center, object_center2], fill=aff_rgb, width=10)
    x, y = object_center[0], object_center[1]
    r=10
    draw.ellipse((x-r, y-r, x+r, y+r), fill=manip_rgb)
    x, y = object_center2[0], object_center2[1]
    draw.ellipse((x-r, y-r, x+r, y+r), fill=aff_rgb)

color_rgb = [(255,255,0), (255, 128,0), (128,255,0), (0,128,255), (0,0,255), (127,0,255), (255,0,255), (255,0,127), (255,0,0), (255,204,153), (255,102,102), (153,255,153), (153,153,255), (0,0,153)]
color_rgba = [(255,255,0,70), (255, 128,0,70), (128,255,0,70), (0,128,255,70), (0,0,255,70), (127,0,255,70), (255,0,255,70), (255,0,127,70), (255,0,0,70), (255,204,153,70), (255,102,102,70), (153,255,153,70), (153,153,255,70), (0,0,153,70)]


hand_rgb = [(0, 90, 181), (220, 50, 32)] 
hand_rgba = [(0, 90, 181, 25), (220, 50, 32, 25)]

# manipulated object
manip_rgb = (255, 194, 10)
manip_rgba = (255, 194, 10, 25)

# affected object
aff_rgb = (181, 255, 20)
aff_rgba = (181, 255, 20, 25)

side_map = {'l':'Left', 'r':'Right'}
side_map2 = {0:'Left', 1:'Right'}
side_map3 = {0:'L', 1:'R'}
state_map = {0:'No Contact', 1:'Self Contact', 2:'Another Person', 3:'Portable Object', 4:'Stationary Object'}
state_map2 = {0:'N', 1:'S', 2:'O', 3:'P', 4:'F'}

