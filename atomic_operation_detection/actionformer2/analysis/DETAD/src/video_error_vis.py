import pandas as pd
import plotly.express as px
import plotly.io as pio
import json
import os
from glob import glob
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

from action_detector_diagnosis import ActionDetectorDiagnosis


# test participants: 3,8,13,20,28
video_id = "P03_05_01"
type_name = "action"
pred_file = "/work/ohashi/AIP/action_detection/actionformer/action/aip_i3d_action30_layer4overlap[0,4]_w19/results/action_eval_results.json"
gt_file = "/work/ohashi/AIP/action_detection/data/annotations/annotation_all.json"
seg_gt_dir = "/work/ohashi/AIP/action_segmentation/data/groundTruth"
image_dir = "/work/ohashi/AIP/frames_256p"
save_dir = f"/work/ohashi/AIP/action_detection/actionformer/action/aip_i3d_action30_layer4overlap[0,4]_w19/analysis/{video_id}"
task_pallete = "/work/ohashi/AIP/scripts/action_segmentation/color_pallete.txt"
tiou = 0.3
hand = "none"
output_video = True


def calc_tIoU(pred, gt):
    tt1 = np.maximum(pred["start"], gt["start"].values)
    tt2 = np.minimum(pred["end"], gt["end"].values)
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (pred["end"] - pred["start"]) \
                     + (gt["end"].values - gt["start"].values) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


os.makedirs(save_dir, exist_ok=True)
fps = json.load(open(gt_file, 'r'))["database"][video_id]["fps"]

fp_error_analysis = ActionDetectorDiagnosis(ground_truth_filename=gt_file,
                                            prediction_filename=pred_file,
                                            tiou_thresholds=[tiou],
                                            min_tiou_thr=0.1,
                                            subset="test",
                                            verbose=False, 
                                            normalize_ap=False,
                                            type=type_name,
                                            hand_annotation=hand
                                        )

fp_error_analysis.evaluate()
fp_error_analysis.diagnose()

column_name = fp_error_analysis.fp_error_type_cols[0]
activity_index_reverse = fp_error_analysis.activity_index_reverse
fp_error_types_legend = fp_error_analysis.fp_error_types_legned
fp_error_types_inverse_legend = fp_error_analysis.fp_error_types_inverse_legned
prediction = fp_error_analysis.prediction
ground_truth = fp_error_analysis.ground_truth
prediction = prediction.groupby("video-id").get_group(video_id)
ground_truth = ground_truth.groupby("video-id").get_group(video_id)

# summarize data to visualize
vis_data = []
for error_id in fp_error_types_inverse_legend:
    error_prediction = prediction[prediction[column_name] == error_id].reset_index(drop=True)
    for idx in range(len(error_prediction)):
        vis_data.append(
            dict(
                type=fp_error_types_inverse_legend[error_id],
                start=error_prediction.iloc[idx]["t-start"], end=error_prediction.iloc[idx]["t-end"],
                label=activity_index_reverse[error_prediction.iloc[idx]["label"]],
            )
        )

for idx in range(len(ground_truth)):
    vis_data.append(
        dict(
            type="Ground Truth",
            start=ground_truth.iloc[idx]["t-start"], end=ground_truth.iloc[idx]["t-end"],
            label=activity_index_reverse[ground_truth.iloc[idx]["label"]],
        )
    )

# read task info
task_file = os.path.join(seg_gt_dir, f"{video_id}.txt")
tasks = open(task_file, 'r').read().split()
cur_task = tasks[0]
start, end = 0, 0
for i, task in enumerate(tasks):
    if task == cur_task:
        continue
    end = i / fps
    vis_data.append(
        dict(
            type="Task",
            start=start, end=end,
            label=cur_task
        )
    )
    start = (i + 1) / fps
    cur_task = task

# create graph 
df = pd.DataFrame(vis_data)
df["delta"] = df["end"] - df["start"]
if task_pallete:
    color_pallete = open(task_pallete, 'r').read().split('\n')
    color_pallete = {x.split()[0] : x.split()[1] for x in color_pallete}
    color_pallete.update({"close": "red", "detach": "blue", "eject": "green", "insert": "yellow", "open": "fuchsia", "press": "orange", "put": "purple", "release": "aqua", "shake": "slategrey", "take": "sienna"})
fig = px.timeline(
    df,  # dataframe
    x_start='start', x_end='end',  # colonm name for horizontal axis 
    y='type',  # column name for vertical axis
    color='label',  # column name for color separation
    color_discrete_sequence=px.colors.qualitative.Light24,
    color_discrete_map=color_pallete,
    opacity=0.7,
    category_orders={"type": ["Task", "Ground Truth", "True Positive", 'Double Detection Err', "Localization Err", "Wrong Label Err", "Confusion Err", "Background Err"]},
    title=f"{video_id} results",
)
fig.layout.xaxis.type = 'linear'
for d in fig.data:
    filt = df.label == d.name
    d.x = df[filt]['delta'].tolist()
fig.update_layout(font_size=20, hoverlabel_font_size=20)
fig.show()

pio.orca.config.executable = '/Applications/orca.app/Contents/MacOS/orca'
pio.write_html(fig, os.path.join(save_dir, f"{video_id}_{type_name}.html"))
# pio.write_image(fig, os.path.join(save_dir, f"{video_id}.png"))

# output video
if output_video:
    def add_text(frame, text, x=0, y=50, color=(255, 255, 255)):
        cv2.putText(frame, text=text, org=(x, y), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.5, color=color, thickness=1, lineType=cv2.LINE_AA)
        return frame

    video_image_dir = os.path.join(image_dir, video_id.split('_')[0], video_id)
    # video
    save_file = os.path.join(save_dir, f"{video_id}_{type_name}.mp4")
    writer = None
    gt = df[df["type"] == "Ground Truth"].sort_values(by="start")
    tp = df[df["type"] == "True Positive"].sort_values(by="start")
    label_er = df[df["type"] == "Wrong Label Err"].sort_values(by="start")  # wrong label
    loc_er = df[df["type"] == "Localization Err"].sort_values(by="start")  # localization error
    conf_er = df[df["type"] == "Confusion Err"].sort_values(by="start")  # confusion error
    bg_er = df[df["type"] == "Background Err"].sort_values(by="start")  # background error
    for fr_num, fname in enumerate(sorted(glob(video_image_dir + '/*'))):
        # read an image
        frame = cv2.imread(fname)
        h, w, _ = frame.shape
        frame = cv2.vconcat([frame, 255 + 0 * frame.copy()[:110]])  # concat space to show labels
        if fr_num == 0:
            writer = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
        # find gt label and prediction label appearing in the frame
        time = fr_num / fps
        this_gt = gt[(gt["start"] <= time) & (time <= gt["end"])]
        this_tp = tp[(tp["start"] <= time) & (time <= tp["end"])]
        this_label_er = label_er[(label_er["start"] <= time) & (time <= label_er["end"])]
        this_loc_er = loc_er[(loc_er["start"] <= time) & (time <= loc_er["end"])]
        this_conf_er = conf_er[(conf_er["start"] <= time) & (time <= conf_er["end"])]
        this_bg_er = bg_er[(bg_er["start"] <= time) & (time <= bg_er["end"])]
        for i in range(len(this_gt)):
            frame = add_text(frame, this_gt.iloc[i]["label"], x=0, y=(h + 15 * (i + 1)), color=(0, 0, 0))
        j = 0
        for i in range(len(this_tp)):
            "TP: blue"
            frame = add_text(frame, this_tp.iloc[i]["label"], x=(w // 2), y=(h + 15 * (i + 1)), color=(255, 0, 0))
            j += 1
        for i in range(len(this_label_er)):
            "Wrong Label: orange"
            frame = add_text(frame, this_label_er.iloc[i]["label"], x=(w // 2), y=(h + 15 * (j + 1)), color=(0, 165, 255))
            j += 1
        for i in range(len(this_loc_er)):
            "Localization: purple"
            frame = add_text(frame, this_loc_er.iloc[i]["label"], x=(w // 2), y=(h + 15 * (j + 1)), color=(164, 87, 148))
            j += 1
        for i in range(len(this_conf_er)):
            
            frame = add_text(frame, this_conf_er.iloc[i]["label"], x=(w // 2), y=(h + 15 * (j + 1)), color=(0, 0, 255))
            j += 1
        for i in range(len(this_bg_er)):
            frame = add_text(frame, this_bg_er.iloc[i]["label"], x=(w // 2), y=(h + 15 * (j + 1)), color=(107, 149, 197))
            j += 1
        writer.write(frame)
    writer.release()
