from action_detector_diagnosis import ActionDetectorDiagnosis

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import pandas as pd
import os
from collections import OrderedDict
from matplotlib import gridspec, rc
import matplotlib as mpl
import matplotlib.font_manager
mpl.use('Agg')
params = {
            'xtick.major.size': 8,
            'ytick.major.size': 8,
            'xtick.major.width': 3,
            'ytick.major.width': 3,
            'mathtext.fontset': 'custom',
        }
mpl.rcParams.update(params)
import matplotlib.pyplot as plt

def fn_distribution_by_characteristic(fn_error_analysis, characteristic_name):
    characteristic_distribution_df = OrderedDict()
    for tidx, tiou in enumerate(fn_error_analysis.tiou_thresholds):
        matched_gt_id = fn_error_analysis.prediction[fn_error_analysis.matched_gt_id_cols[tidx]]
        unmatched_gt = fn_error_analysis.ground_truth[~fn_error_analysis.ground_truth['gt-id'].isin(matched_gt_id)]
        all_gt_value, all_gt_count = np.unique(fn_error_analysis.ground_truth[characteristic_name],return_counts=True)
        unmatched_gt_value, unmatched_gt_count = np.unique(unmatched_gt[characteristic_name],return_counts=True)

        characteristic_distribution = OrderedDict()
        sum_counts = all_gt_count.sum()
        for v, c in zip(all_gt_value, all_gt_count):
            characteristic_distribution[v] = {'all': c/sum_counts, 'unmatched': 0}
        for v, c in zip(unmatched_gt_value, unmatched_gt_count):
            characteristic_distribution[v]['unmatched'] = c / (characteristic_distribution[v]['all']*sum_counts)

        characteristic_distribution_df[tiou] = pd.DataFrame(characteristic_distribution).T.fillna(0)
    
    x = list(characteristic_distribution_df.values())
    characteristic_distribution_df_mean = x[0].copy()
    for this_characteristic_distribution_df in x[1:]:
        characteristic_distribution_df_mean += this_characteristic_distribution_df
    characteristic_distribution_df['avg'] = characteristic_distribution_df_mean / len(characteristic_distribution_df)

    return characteristic_distribution_df

def fn_distribution_by_pairwaise_characteristics(fn_error_analysis, characteristic_name_1, characteristic_name_2):
    characteristic_distribution_df = OrderedDict()
    for tidx, tiou in enumerate(fn_error_analysis.tiou_thresholds):
        matched_gt_id = fn_error_analysis.prediction[fn_error_analysis.matched_gt_id_cols[tidx]]
        unmatched_gt = fn_error_analysis.ground_truth[~fn_error_analysis.ground_truth['gt-id'].isin(matched_gt_id)]
        all_gt_value, all_gt_count = [], []
        for group, this_group_gt in fn_error_analysis.ground_truth.groupby([characteristic_name_1, characteristic_name_2]):
            all_gt_value.append(group)
            all_gt_count.append(len(this_group_gt))
        unmatched_gt_value, unmatched_gt_count = [], []
        for group, this_group_gt in unmatched_gt.groupby([characteristic_name_1, characteristic_name_2]):
            unmatched_gt_value.append(group)
            unmatched_gt_count.append(len(this_group_gt))
        all_gt_count = np.array(all_gt_count)
        unmatched_gt_count = np.array(unmatched_gt_count)

        characteristic_distribution = OrderedDict()
        sum_counts = all_gt_count.sum()
        for v, c in zip(all_gt_value, all_gt_count):
            characteristic_distribution[v] = {'all': c/sum_counts, 'unmatched': 0}
        for v, c in zip(unmatched_gt_value, unmatched_gt_count):
            characteristic_distribution[v]['unmatched'] = c / (characteristic_distribution[v]['all']*sum_counts)

        characteristic_distribution_df[tiou] = pd.DataFrame(characteristic_distribution).T.fillna(0)
    
    x = list(characteristic_distribution_df.values())
    characteristic_distribution_df_mean = x[0].copy()
    for this_characteristic_distribution_df in x[1:]:
        characteristic_distribution_df_mean += this_characteristic_distribution_df
    characteristic_distribution_df['avg'] = characteristic_distribution_df_mean / len(characteristic_distribution_df)

    return characteristic_distribution_df

def plot_fn_analysis(fn_error_analysis, save_filename, 
                     colors=['#7fc97f','#beaed4','#fdc086','#386cb0','#f0027f','#bf5b17'],
                     characteristic_names=['context-size', 'context-distance', 'agreement','coverage', 'length', 'num-instances'],
                     characteristic_names_in_text=['Context Size', 'Context Distance', 'Agreement', 'Coverage', 'Length', '\# Instances'],
                     characteristic_names_delta_positions=[1.25,-0.9,0.45,0.7,1,-0.1],
                     buckets_order=['0','1','2','3','4','5','6','XW', 'W', 'XS','S', 'N', 'M', 'F', 'Inf', 'L', 'XL', 'H', 'XH'],
                     figsize=(20,3.5), fontsize=24):

    # characteristic distribution
    characteristic_distribution = OrderedDict()
    for characteristic_name in characteristic_names:
        characteristic_distribution[characteristic_name] = fn_distribution_by_characteristic(fn_error_analysis, characteristic_name)

    characteristic_name_lst,bucket_lst,ratio_value_lst = [],[],[]
    for characteristic_name in characteristic_names:
        values = characteristic_distribution[characteristic_name]['avg']['unmatched'].values
        xticks = characteristic_distribution[characteristic_name]['avg'].index
        for i in range(len(values)):
            characteristic_name_lst.append(characteristic_name)
            bucket_lst.append(xticks[i])
            ratio_value_lst.append(values[i])

    # characteristic-name,bucket,ratio-value
    false_negative_rate_df = pd.DataFrame({'characteristic-name': characteristic_name_lst,
                                           'bucket': bucket_lst,
                                           'ratio-value': ratio_value_lst,
                                          })
    false_negative_rate_df['order'] = pd.Categorical(false_negative_rate_df['bucket'],
                                                     categories=buckets_order,ordered=True)
    false_negative_rate_df.sort_values(by='order', inplace=True)
    false_negative_rate_df_by_characteristic_name = false_negative_rate_df.groupby('characteristic-name')

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    current_x_value = 0
    xticks_lst,xvalues_lst = [],[]
    for char_idx, characteristic_name in enumerate(characteristic_names):
        this_false_negative_rate = false_negative_rate_df_by_characteristic_name.get_group(characteristic_name)
        x_values = range(current_x_value, current_x_value + len(this_false_negative_rate))
        y_values = this_false_negative_rate['ratio-value'].values*100
        mybars = plt.bar(x_values, y_values, color=colors[char_idx])
        for bari in mybars:
            height = bari.get_height()
            plt.gca().text(bari.get_x() + bari.get_width()/2, bari.get_height()+0.025*100, '%.1f' % height,
                         ha='center', color='black', fontsize=fontsize/1.15)
        ax.annotate(characteristic_names_in_text[char_idx],
                    xy=(current_x_value+characteristic_names_delta_positions[char_idx],100),
                    fontsize=fontsize)

        if char_idx < len(characteristic_names) - 1:
            ax.axvline(max(x_values)+1, linewidth=1.5, color="gray", linestyle='dotted')

        current_x_value = max(x_values) + 2
        xticks_lst.extend(this_false_negative_rate['bucket'].values.tolist())
        xvalues_lst.extend(x_values)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='dotted')
    ax.set_axisbelow(True)
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(size=10, direction='in', width=2)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(2.5)
    plt.xticks(xvalues_lst, xticks_lst, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('False Negative $(\%)$', fontsize=fontsize)
    plt.tight_layout()
    plt.ylim(0,1.1*100)
    fig.savefig(save_filename, bbox_inches='tight')
    print('[Done] Output analysis is saved in %s' % save_filename)

def main(ground_truth_filename, subset, prediction_filename, output_folder, type, hand_annotation, tiou_thresholds):
    os.makedirs(output_folder, exist_ok=True)

    # characteristic_names_to_bins = {'coverage': (np.array([0,0.002,0.003,0.004,0.006,1]), ['XS','S','M','L','XL']),
    #                                 'length': (np.array([0,0.4,0.6,0.8,1.0,np.inf]), ['XS','S','M','L','XL']),
    #                                 'num-instances': (np.array([-1,200,220,250,np.inf]), ['XS','S','M','L'])}
    characteristic_names_to_bins = {'length': (np.array([0,0.4099999999999966,0.6299999999999955, 0.8700000000000045,1.2300000000000182,np.inf]), ['XS','S','M','L','XL']),
                                    'num-instances': (np.array([-1,180,222,259,296,np.inf]), ['XS','S','M','L', 'XL'])}
    # colors = ['#386cb0','#f0027f','#bf5b17']
    colors = ['#f0027f','#bf5b17']
    characteristic_names = ['length', 'num-instances']
    characteristic_names_in_text = ['Length', 'Instances']
    # characteristic_names_delta_positions = [0.5,1,-0.2]
    characteristic_names_delta_positions = [1.3,1]
    # figsize = (10,3.5)
    figsize = (8,4)

    fn_error_analysis = ActionDetectorDiagnosis(ground_truth_filename=ground_truth_filename,
                                                prediction_filename=prediction_filename,
                                                tiou_thresholds=np.array(tiou_thresholds),
                                                limit_factor=10,
                                                min_tiou_thr=0.1,
                                                subset=subset, 
                                                verbose=True, 
                                                load_extra_annotations=True,
                                                characteristic_names_to_bins=characteristic_names_to_bins,
                                                normalize_ap=False,
                                                minimum_normalized_precision_threshold_for_detection=0.05,
                                                type=type,
                                                hand_annotation=hand_annotation
                                            )

    fn_error_analysis.evaluate()

    plot_fn_analysis(fn_error_analysis=fn_error_analysis,
                     save_filename=os.path.join(output_folder, f'{type}_tiou{"-".join(map(str, tiou_thresholds))}_false_negative_analysis.pdf'),
                     colors=colors,
                     characteristic_names=characteristic_names,
                     characteristic_names_in_text=characteristic_names_in_text,
                     characteristic_names_delta_positions=characteristic_names_delta_positions,
                     figsize=figsize,
                     fontsize=16)

if __name__ == '__main__':
    parser = ArgumentParser(description='Run the false negative error analysis.',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ground_truth_filename', required=True, type=str,
                        help='The path to the JSON file containing the ground truth annotations')
    parser.add_argument('--subset', default='test', type=str,
                        help='The dataset subset to use for the analysis')
    parser.add_argument('--prediction_filename', required=True, type=str,
                        help='The path to the JSON file containing the method\'s predictions')
    parser.add_argument('--output_folder', required=True, type=str,
                        help='The path to the folder in which the results will be saved')
    parser.add_argument('--type', required=True, type=str,
                        help='annotationtype (verb, manipulated, effect, action)')
    parser.add_argument('--hand', type=str, default="none",
                        help='hand annotation to use (none, left, right)')
    parser.add_argument('--tiou_thresholds', type=float, nargs="*", default='0.3',
                        help='tiou thresholds used to take mAP average over')
    args = parser.parse_args()
    if type(args.tiou_thresholds) == float:
        args.tiou_thresholds = [args.tiou_thresholds]

    main(args.ground_truth_filename, args.subset, args.prediction_filename, args.output_folder, args.type, args.hand, args.tiou_thresholds)
