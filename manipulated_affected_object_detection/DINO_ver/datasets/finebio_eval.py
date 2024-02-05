import pandas as pd
import numpy as np
import copy


def convert_xywh_to_xyxy(box):
    box[2:] = box[:2] + box[2:]
    return box
  
    
def calc_iou(pred_box, gt_boxes):
    ixmin = np.maximum(gt_boxes[:, 0], pred_box[0])
    iymin = np.maximum(gt_boxes[:, 1], pred_box[1])
    ixmax = np.minimum(gt_boxes[:, 2], pred_box[2])
    iymax = np.minimum(gt_boxes[:, 3], pred_box[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih
    
    uni = (pred_box[2] - pred_box[0] + 1.) * (pred_box[3] - pred_box[1] + 1.) + \
        (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.) -inters
        
    ious = inters / uni
    return ious


def calc_ap(rec, prec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def finebio_eval(result_df, gt_df, class_name, ovthresh=0.5):
    try:
        cls_gt_df = gt_df.groupby("class_name").get_group(class_name).reset_index(drop=True)
    except Exception as e:
        # when there's no ground truth for the class, ap is 0
        return None, None, 0
    
    try:
        cls_result_df = result_df.groupby("class_name").get_group(class_name).reset_index(drop=True)
    except Exception as e:
        # when there's no prediction for the class, ap is 0
        return None, None, 0
    
    npos = cls_gt_df.shape[0]
    det = [False] * npos
    npred = cls_result_df.shape[0]
    tp = np.zeros(npred)
    fp = np.zeros(npred)
    
    cls_gt_df_byid = cls_gt_df.groupby("image_id")
    cls_result_df = cls_result_df.sort_values(by="score", ascending=False)
    cls_result_df = cls_result_df.reset_index(drop=True)
    for idx, this_pred in cls_result_df.iterrows():
        try:
            # Check if there is at least one ground truth in the video associated.
            this_gt = cls_gt_df_byid.get_group(this_pred.image_id)
        except Exception as e:
            fp[idx] = 1.
            continue
        this_gt = this_gt.reset_index()
        ious = calc_iou(this_pred[["x1", "y1", "x2", "y2"]].values, this_gt[["x1", "y1", "x2", "y2"]].values)
        ioumax = np.max(ious)
        max_ind = np.argmax(ious)
        
        if ioumax > ovthresh:
            if not det[this_gt.loc[max_ind]['index']]:
                tp[idx] = 1.
                det[this_gt.loc[max_ind]['index']] = True
            else:
                fp[idx] = 1.
        else:
            fp[idx] = 1.
    
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = calc_ap(rec, prec)
    
    return rec, prec, ap


def calc_center(box):
    return [(box[0] + box[2]) / 2., (box[1] + box[3]) / 2.]


def get_hand_with_manipulated(result_df, hand_name):
    # extract results of hands predicted as hand_name
    hand_result_df = result_df[result_df["class_name"] == hand_name].reset_index(drop=True)
    # extract results of non-hand objects predicted as 'manipulated'
    manip_result_df = result_df[(result_df["manipulatedstate"] == 1) & (result_df["class_name"] != "left_hand") & (result_df["class_name"] != "right_hand")].reset_index(drop=True)
    
    # if there's no hand or no possibly manipulated object, no need to do anything.
    if hand_result_df.shape[0] == 0 or manip_result_df.shape[0] == 0:
        return hand_result_df
    
    # calculate manipulated object candidates' centers
    manip_boxes = manip_result_df[["x1", "y1", "x2", "y2"]].values
    manip_centers = np.apply_along_axis(lambda x: calc_center(x), 1, manip_boxes)
  
    hand_result_df = hand_result_df.sort_values(by='image_id').reset_index(drop=True)
    manip_result_df = manip_result_df.groupby('image_id')
    pre_img_id, this_manip = None, None
    for idx, this_hand in hand_result_df.iterrows():
        # if hand state is predicted as 'non-manipulating', no manipulated object by the hand.
        if this_hand.handstate == 0:
            continue
        cur_img_id = this_hand.image_id
        if cur_img_id != pre_img_id:
            # Check if there is at least one possibly manipulated in the video associated.
            try:
                this_manip = manip_result_df.get_group(cur_img_id).reset_index()
            except Exception as e:
                continue
            pre_img_id = cur_img_id
        # Calculate a point the hand points at.
        hand_center = calc_center(this_hand[["x1", "y1", "x2", "y2"]].values)
        hand_dest = np.array([hand_center[0] + this_hand["dx"] * this_hand["magnitude"], hand_center[1] + this_hand["dy"] * this_hand["magnitude"]])
        dist = np.sum((manip_centers[this_manip['index'].values] - hand_dest)**2, 1)
        min_ind = np.argmin(dist)
        hand_result_df.loc[idx, 'obj_class_name'] = this_manip.loc[min_ind, 'class_name']
        hand_result_df.loc[idx, 'obj_id'] = this_manip.loc[min_ind, 'id']
        hand_result_df.loc[idx, ["obj_x1", "obj_y1", "obj_x2", "obj_y2"]] = manip_boxes[this_manip.loc[min_ind, 'index']]
    return hand_result_df


def val_pointed_object(pred, gt, threshold=0.5, class_match=True):
    # if prediction and gt both have no manipulated/affected object, return True.
    # okay to just check object1 in gt, since the following objects are all None if the first is None.
    if pd.isnull(pred['obj_class_name']) and pd.isnull(gt['obj_class_name1']):
        return (True, False, False)
    # if gt has any manipulated/affected object, predicted manipulated/affected object should be not None.
    if pd.isnull(pred["obj_class_name"]):
        return (False, False, False)
    # if gt has no manipulated/affected object, predicted manipulated/affected object should be None.
    if pd.isnull(gt['obj_class_name1']):
        return (False, False, False)
    # check if a predicted manipulated/affected object matches gt.
    # if class_match is False, just compare bbox. Otherwise, compare both bbox and class name.
    obj1_match, obj2_match, obj3_match = False, False, False
    iou = calc_iou(pred[["obj_x1", "obj_y1", "obj_x2", "obj_y2"]].values, gt[["obj1_x1", "obj1_y1", "obj1_x2", "obj1_y2"]].values[None, :])[0]
    obj1_match = iou > threshold and pred['obj_class_name'] == gt['obj_class_name1'] if class_match else iou > threshold
    if not pd.isnull(gt['obj_class_name2']):
        iou = calc_iou(pred[["obj_x1", "obj_y1", "obj_x2", "obj_y2"]].values, gt[["obj2_x1", "obj2_y1", "obj2_x2", "obj2_y2"]].values[None, :])[0]
        obj2_match = iou > threshold and pred['obj_class_name'] == gt['obj_class_name2'] if class_match else iou > threshold
    if not pd.isnull(gt['obj_class_name3']):
        iou = calc_iou(pred[["obj_x1", "obj_y1", "obj_x2", "obj_y2"]].values, gt[["obj3_x1", "obj3_y1", "obj3_x2", "obj3_y2"]].values[None, :])[0]
        obj3_match = iou > threshold and pred['obj_class_name'] == gt['obj_class_name2'] if class_match else iou > threshold
    return [obj1_match, obj2_match, obj3_match]


def get_manip_with_affected(result_df, hand_name):
    # Extract results of non-hand objects predicted as 'manipulated'
    manip_result_df = result_df[(result_df["manipulatedstate"] == 1) & (result_df["class_name"] != "left_hand") & (result_df["class_name"] != "right_hand")].reset_index(drop=True)
    # Extract results of objects which is different from hand_name and predicted as 'affected'
    affect_result_df = result_df[(result_df["affectedstate"] == 1) & (result_df["class_name"] != hand_name)].reset_index(drop=True)
    
    if manip_result_df.shape[0] == 0 or affect_result_df.shape[0] == 0:
        return manip_result_df
    
    # calculate affected object candidates' centers
    affect_boxes = affect_result_df[["x1", "y1", "x2", "y2"]].values
    affect_centers = np.apply_along_axis(lambda x: calc_center(x), 1, affect_boxes)
    
    manip_result_df = manip_result_df.sort_values(by='image_id').reset_index(drop=True)
    affect_result_df = affect_result_df.groupby('image_id')
    pre_img_id, this_affect = None, None
    for idx, this_manip in manip_result_df.iterrows():
        # if the manipulated object is predicted as 'non-affecting', no affected object by the object.
        if this_manip.affectingstate == 0:
            continue
        cur_img_id = this_manip.image_id
        if cur_img_id != pre_img_id:
            # check if there is at least one possibly affected object in the video associated.
            try:
                this_affect = affect_result_df.get_group(cur_img_id)
            except Exception as e:
                continue
            pre_img_id = cur_img_id
        # calculate a point the manipulated object points at.
        this_affect_rm = this_affect[this_affect["id"] != this_manip.id].reset_index()  # remove manipulated object from candidates
        if len(this_affect_rm) == 0:
            continue
        manip_center = calc_center(this_manip[["x1", "y1", "x2", "y2"]].values)
        manip_dest = np.array([manip_center[0] + this_manip["dx"] * this_manip["magnitude"], manip_center[1] + this_manip["dy"] * this_manip["magnitude"]])
        dist = np.sum((affect_centers[this_affect_rm['index'].values] - manip_dest)**2, 1)
        min_ind = np.argmin(dist)
        manip_result_df.loc[idx, 'obj_id'] = this_affect_rm.loc[min_ind, 'id']
        manip_result_df.loc[idx, 'obj_class_name'] = this_affect_rm.loc[min_ind, 'class_name']
        manip_result_df.loc[idx, ["obj_x1", "obj_y1", "obj_x2", "obj_y2"]] = affect_boxes[this_affect_rm.loc[min_ind, 'index']]
    return manip_result_df


def val_affected_object(pred, gt, manip_result_df, gt_df, threshold=0.5, class_match=True):
    # first check the manipulated object match.
    manip1_match, manip2_match, manip3_match = val_pointed_object(pred, gt)
    # when manipulated objects don't match.
    if manip1_match == False and manip2_match == False and manip3_match == False:
        return False
    # when manipulated object doesn't exist and it's predicted correctly.
    if manip1_match and pd.isnull(pred['obj_class_name']):
        return True
    # when predicted manipulated object matches at least one of gt manipulated objects,
    # check if affected object also matches.
    ret = False
    if manip1_match:
        ret = ret or np.any(val_pointed_object(manip_result_df[manip_result_df["id"] == pred["obj_id"]].iloc[0], 
                                  gt_df[gt_df["id"]==gt["obj_id1"]].iloc[0], threshold=threshold, class_match=class_match))
    if (not ret) and manip2_match:
        ret = ret or np.any(val_pointed_object(manip_result_df[manip_result_df["id"] == pred["obj_id"]].iloc[0], 
                                  gt_df[gt_df["id"]==gt["obj_id2"]].iloc[0], threshold=threshold, class_match=class_match))
    if (not ret) and manip3_match:
        ret = ret or np.any(val_pointed_object(manip_result_df[manip_result_df["id"] == pred["obj_id"]].iloc[0], 
                                  gt_df[gt_df["id"]==gt["obj_id3"]].iloc[0], threshold=threshold, class_match=class_match))
    return ret


def finebio_eval_hand(result_df, gt_df, class_name, ovthresh=0.5, constraint=''):
    try:
        hand_gt_df = gt_df.groupby("class_name").get_group(class_name).reset_index(drop=True)
    except Exception as e:
        # when there's no ground truth for the class(hand), ap is 0
        return None, None, 0
    
    hand_result_df = get_hand_with_manipulated(result_df, class_name)
    if hand_result_df.shape[0] == 0:
        # when there's no prediction for the class, ap is 0
        return None, None, 0
    hand_result_df.to_csv(f"{class_name}_manipulated.csv", index=False)
    
    if constraint in ['manipulated_bbox&class_affected_bbox', 'manipulated_bbox&class_affected_bbox&class']:
        manip_result_df = get_manip_with_affected(result_df, class_name)
        manip_result_df.to_csv(f"{class_name}_affected.csv", index=False)
    
    npos = hand_gt_df.shape[0]
    det = [False] * npos
    npred = hand_result_df.shape[0]
    tp = np.zeros(npred)
    fp = np.zeros(npred)
    
    hand_gt_df_byid = hand_gt_df.groupby("image_id")
    hand_result_df = hand_result_df.sort_values(by="score", ascending=False)
    hand_result_df = hand_result_df.reset_index(drop=True)
    for idx, this_pred in hand_result_df.iterrows():
        try:
            # Check if there is at least one ground truth in the video associated.
            this_gt = hand_gt_df_byid.get_group(this_pred.image_id)
        except Exception as e:
            fp[idx] = 1.
            continue
        this_gt = this_gt.reset_index()
        ious = calc_iou(this_pred[["x1", "y1", "x2", "y2"]].values, this_gt[["x1", "y1", "x2", "y2"]].values)
        ioumax = np.max(ious)
        max_ind = np.argmax(ious)
        
        if constraint == '':
            if ioumax > ovthresh:
                if not det[this_gt.loc[max_ind]['index']]:
                    tp[idx] = 1.
                    det[this_gt.loc[max_ind]['index']] = True
                else:
                    fp[idx] = 1.
            else:
                fp[idx] = 1.
        
        elif constraint == 'handstate':
            if ioumax > ovthresh:
                if not det[this_gt.loc[max_ind]['index']] and this_pred['handstate'] == this_gt.loc[max_ind]['handstate']:
                    tp[idx] = 1.
                    det[this_gt.loc[max_ind]['index']] = True
                else:
                    fp[idx] = 1.
            else:
                fp[idx] = 1.
        
        elif constraint == 'manipulated_bbox':
            if ioumax > ovthresh:
                if not det[this_gt.loc[max_ind]['index']] and \
                    np.any(val_pointed_object(this_pred, this_gt.loc[max_ind], class_match=False)):
                    tp[idx] = 1.
                    det[this_gt.loc[max_ind]['index']] = True
                else:
                    fp[idx] = 1.
            else:
                fp[idx] = 1.  
        
        elif constraint == 'manipulated_bbox&class':
            if ioumax > ovthresh:
                if not det[this_gt.loc[max_ind]['index']] and \
                    np.any(val_pointed_object(this_pred, this_gt.loc[max_ind])):
                    tp[idx] = 1.
                    det[this_gt.loc[max_ind]['index']] = True
                else:
                    fp[idx] = 1.
            else:
                fp[idx] = 1. 
        
        elif constraint == 'manipulated_bbox&class_affected_bbox':
            if ioumax > ovthresh:
                if not det[this_gt.loc[max_ind]['index']] and \
                    val_affected_object(this_pred, this_gt.loc[max_ind], manip_result_df, gt_df, class_match=False):
                    tp[idx] = 1.
                    det[this_gt.loc[max_ind]['index']] = True
                else:
                    fp[idx] = 1.
            else:
                fp[idx] = 1.
        
        elif constraint == 'manipulated_bbox&class_affected_bbox&class':
            if ioumax > ovthresh:
                if not det[this_gt.loc[max_ind]['index']] and \
                    val_affected_object(this_pred, this_gt.loc[max_ind], manip_result_df, gt_df):
                    tp[idx] = 1.
                    det[this_gt.loc[max_ind]['index']] = True
                else:
                    fp[idx] = 1.
            else:
                fp[idx] = 1.
    
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = calc_ap(rec, prec)
    
    return rec, prec, ap


class HandObjEvaluator(object):
    def __init__(self, coco_gt, eval_constraints=['handstate', 'manipulated_bbox', 'manipulated_bbox&class', 'manipulated_bbox&class_affected_bbox', 'manipulated_bbox&class_affected_bbox&class']):
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.classes = tuple(['__background__'] + [c['name'] for c in self.coco_gt.loadCats(self.coco_gt.getCatIds())])
        self.gt_dict = {"id": [], "image_id": [], "class_name": [], "x1": [], "y1": [], "x2":[], "y2": [], 
                        "handstate": [], "manipulatedstate": [], "affectingstate": [], "affectedstate": [],
                        "magnitude1": [], "dx1": [], "dy1": [], "obj_id1": [],
                        "obj_class_name1": [], "obj1_x1": [], "obj1_y1": [], "obj1_x2": [], "obj1_y2": [],
                        "magnitude2": [], "dx2": [], "dy2": [], "obj_id2": [],
                        "obj_class_name2": [], "obj2_x1": [], "obj2_y1": [], "obj2_x2": [], "obj2_y2": [],
                        "magnitude3": [], "dx3": [], "dy3": [], "obj_id3": [],
                        "obj_class_name3": [], "obj3_x1": [], "obj3_y1": [], "obj3_x2": [], "obj3_y2": []}
        self.result_dict = {"image_id": [], "class_name": [], "x1": [], "y1": [], "x2":[], "y2": [], 
                        "score": [], "handstate": [], "manipulatedstate": [], "affectingstate": [], "affectedstate": [], 
                        "magnitude": [], "dx": [], "dy": [], "obj_id": [],
                        "obj_class_name": [], "obj_x1": [], "obj_y1": [], "obj_x2": [], "obj_y2": []}
        self.eval_constraints = eval_constraints
        
    def update_gt_dict(self, target):
        for i in range(len(target['boxes'])):
            ann_id = target["ann_ids"][i]
            self.gt_dict["id"].append(ann_id)
            self.gt_dict['image_id'].append(target['image_id'][0])
            self.gt_dict['class_name'].append(self.classes[target['labels'][i]])
            box = np.array(self.coco_gt.loadAnns([ann_id])[0]["bbox"])  # (x,y,w,h)
            box = convert_xywh_to_xyxy(box)  # (x,y,x,y)
            self.gt_dict['x1'].append(box[0])
            self.gt_dict['y1'].append(box[1])
            self.gt_dict['x2'].append(box[2])
            self.gt_dict['y2'].append(box[3])
            self.gt_dict['handstate'].append(target["hand_states"][i])
            self.gt_dict['manipulatedstate'].append(target["manipulated_states"][i])
            self.gt_dict['affectingstate'].append(target["affecting_states"][i])
            self.gt_dict['affectedstate'].append(target["affected_states"][i])
            for k in range(3):
                self.gt_dict[f'magnitude{k + 1}'].append(target["magnitudes"][i, k] * 1000)
                self.gt_dict[f'dx{k + 1}'].append(target["unitdxs"][i, k])
                self.gt_dict[f'dy{k + 1}'].append(target["unitdys"][i, k])
                obj_id = np.nan if target["dest_ids"][i, k] == -1 else target["dest_ids"][i, k]
                self.gt_dict[f'obj_id{k + 1}'].append(obj_id)
                if not np.isnan(obj_id):
                    dest_obj_info = self.coco_gt.loadAnns([obj_id])[0]
                obj_class_name = np.nan if np.isnan(obj_id) else self.classes[dest_obj_info['category_id']]
                self.gt_dict[f'obj_class_name{k + 1}'].append(obj_class_name)
                obj_box = (np.nan, np.nan, np.nan, np.nan) if np.isnan(obj_id) else convert_xywh_to_xyxy(np.array(dest_obj_info['bbox']))
                self.gt_dict[f'obj{k + 1}_x1'].append(obj_box[0])
                self.gt_dict[f'obj{k + 1}_y1'].append(obj_box[1])
                self.gt_dict[f'obj{k + 1}_x2'].append(obj_box[2])
                self.gt_dict[f'obj{k + 1}_y2'].append(obj_box[3])
                    
    def update_result_dict(self, image_id, cls_id, cls_dets):
       for det in cls_dets:
            self.result_dict['image_id'].append(image_id)
            self.result_dict['class_name'].append(self.classes[cls_id])
            self.result_dict['x1'].append(det[0])
            self.result_dict['y1'].append(det[1])
            self.result_dict['x2'].append(det[2])
            self.result_dict['y2'].append(det[3])
            self.result_dict['score'].append(det[4])
            self.result_dict['handstate'].append(det[5])
            self.result_dict['magnitude'].append(det[6] * 1000)
            self.result_dict['dx'].append(det[7])
            self.result_dict['dy'].append(det[8])
            self.result_dict['manipulatedstate'].append(det[9])
            self.result_dict['affectingstate'].append(det[10])
            self.result_dict['affectedstate'].append(det[11])
            self.result_dict['obj_id'].append(np.nan)
            self.result_dict['obj_class_name'].append(np.nan)
            self.result_dict['obj_x1'].append(np.nan)
            self.result_dict['obj_y1'].append(np.nan)
            self.result_dict['obj_x2'].append(np.nan)
            self.result_dict['obj_y2'].append(np.nan) 
    
    def _do_detection_eval(self, result_df, gt_df):
        aps = []
        hand_aps = []
        for class_name in self.classes:
            if class_name == '__background__' or (class_name not in np.unique(gt_df["class_name"].values)):  # ignore background
                continue
            rec, prec, ap = finebio_eval(result_df, gt_df, class_name, ovthresh=0.5)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(class_name, ap))

            if class_name == "left_hand" or class_name == "right_hand":
                for constraint in self.eval_constraints:
                    rec, prec, ap = finebio_eval_hand(result_df, gt_df, class_name, ovthresh=0.5, constraint=constraint)
                    print('AP for {} + {} = {:.4f}'.format(class_name, constraint, ap))
                hand_aps += [ap]
            
        print('Mean Detection AP = {:.4f}'.format(np.mean(aps)))
        print('Mean Detection+Hand-interaction AP = {:.4f}'.format(np.mean(aps + hand_aps)))
        return np.mean((aps + hand_aps))
        
    def evaluate_detections(self):
        result_df = pd.DataFrame.from_dict(self.result_dict, orient='columns')
        result_df["id"] = np.arange(len(result_df))
        gt_df = pd.DataFrame.from_dict(self.gt_dict, orient="columns")
        all_avg_ap = self._do_detection_eval(result_df, gt_df)
        return all_avg_ap
        
    
