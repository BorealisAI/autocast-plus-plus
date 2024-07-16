# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import json
from tqdm import tqdm
import datetime
import numpy as np

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
PROJ_DIR = os.path.abspath(os.path.join(CUR_DIR, "../.."))
import sys
sys.path.append(PROJ_DIR)
from autocast_experiments.retriever_utils.retriever_common_utils import create_logger


GLBOAL_CUTOFF_DATE='2021-05-11'


def load_data(data_path):
    """
    Load the json question data from the given path
    """
    assert data_path.endswith('.json')
    with open(data_path, 'r') as fin:
        data = json.load(fin)

    global_cutoff_date = datetime.datetime.strptime(GLBOAL_CUTOFF_DATE, '%Y-%m-%d')
    examples = []
    for k, example in tqdm(enumerate(data), desc="Loading data..."):
        del example['background']
        del example['tags']
        del example['source_links']

        publish_time = datetime.datetime.strptime(example['publish_time'][:19], '%Y-%m-%d %H:%M:%S')
        close_time = datetime.datetime.strptime(example['close_time'][:19], '%Y-%m-%d %H:%M:%S')
        total_minutes = (close_time - publish_time).total_seconds() / 60

        qtype = example['qtype']

        # skip questions in the testing split or are potentially unresolved or numerical prediction questions
        if close_time > global_cutoff_date or example['status'] != "Resolved" or qtype == 'num':
            continue

        for i_crowd, crowd_forecast in enumerate(example['crowd']):
            crowd_timestamp = datetime.datetime.strptime(crowd_forecast['timestamp'][:19], '%Y-%m-%d %H:%M:%S')
            minutes_after_publish = (crowd_timestamp - publish_time).total_seconds() / 60
            normalized_time = minutes_after_publish / total_minutes
            normalized_time = np.clip(normalized_time, 0, 1)
            example['crowd'][i_crowd]['normalized_time'] = normalized_time

            if qtype == 't/f':
                # forecast value means the empirical likelihood of the answer being true, it's not the forecast accuracy!
                if example['answer'] == 'yes':
                    forecast_accuracy = crowd_forecast['forecast']
                elif example['answer'] == 'no':
                    forecast_accuracy = 1 - crowd_forecast['forecast']
                else:
                    raise ValueError(f"Unknown answer: {example['answer']} for t/f type question.")
            elif qtype == 'mc':
                answer_index = ord(example['answer']) - ord('A')  # index of the correct answer (e.g., A/B/C/D -> 0/1/2/3)
                forecast_accuracy = crowd_forecast['forecast'][answer_index]
            elif qtype == 'num':
                raise ValueError(f"Unsupported question type: {example['qtype']}")
            else:
                raise ValueError(f"Unknown question type: {example['qtype']}")
            
            # add the forecast accuracy to the crowd forecast
            example['crowd'][i_crowd]['forecast_accuracy'] = forecast_accuracy
        examples.append(example)

    return examples


def adaptive_interpolate(seq, num_points):
    """
    To adaptively interpolate the sequence to a fixed number of points
    """
    time, accuracy = seq[0], seq[1]
    new_time = np.linspace(0, 1, num_points)
    new_accuracy = np.interp(new_time, time, accuracy)
    return new_accuracy


def main():
    # init
    log_file = 'log_recency_rerank_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logger = create_logger(log_file=os.path.join(CUR_DIR, log_file))

    recency_rerank_file = os.path.join(CUR_DIR, '../data/recency_rerank_plot.npz')
    if os.path.exists(recency_rerank_file):
        logger.info(f"Recency rerank data already exists at {recency_rerank_file}, skipping the generation step.")
        recency_rerank = np.load(recency_rerank_file)
        trim_acc_ls = recency_rerank['trim_acc_ls']
        trim_time = recency_rerank['trim_time']
        mean_acc = recency_rerank['mean_acc']
        acc_grad_to_time = recency_rerank['acc_grad_to_time']    
        time_smooth = recency_rerank['time_smooth']    
        acc_grad_to_time_smooth = recency_rerank['acc_grad_to_time_smooth']
    else:
        logger.info(f"Recency rerank data will be saved at {recency_rerank_file}. Start processing...")

        # load the json file for the questions
        data_json = os.path.join(PROJ_DIR, 'autocast_experiments/data/autocast_questions.json')
        questions_with_crowd_acc = load_data(data_json)

        # load crowd forecast accuracy w.r.t. normalized time from all data points
        time_acc_list = []
        for example in questions_with_crowd_acc:
            all_time = [example['crowd'][i]['normalized_time'] for i in range(len(example['crowd']))]
            all_acc = [example['crowd'][i]['forecast_accuracy'] for i in range(len(example['crowd']))]
            time_acc_list.append(np.array([all_time, all_acc]))
        
        stats_num_points = [time_acc_list[i].shape[-1] for i in range(len(time_acc_list))]  # [B]
        logger.info("#crowd forecasts per question statistics: mean: %.2f, std: %.2f, median: %.2f, min: %d, max: %d" % (np.mean(stats_num_points), np.std(stats_num_points), np.median(stats_num_points), np.min(stats_num_points), np.max(stats_num_points)))
        logger.info("quantiles: 0.1%: {:.2f}, 1%: {:.2f}, 5%: {:.2f}, 95%: {:.2f}, 99%: {:.2f}, 99.9%: {:.2f}".format(*np.percentile(stats_num_points, [0.1, 1, 5, 95, 99, 99.9])))

        # apply interpolation and aggregation to the crowd forecast accuracy
        num_points = 109
        trim_time = np.linspace(0, 1, num_points)  # [N]
        trim_acc_ls = []
        for seq in time_acc_list:
            trim_acc_ls.append(adaptive_interpolate(seq, num_points))
        trim_acc_ls = np.array(trim_acc_ls)

        mean_acc = np.mean(trim_acc_ls, axis=0)  # [N]
        acc_grad_to_time = np.gradient(mean_acc, trim_time)  # [N]

        start_idx = int(len(trim_time) * 0.05)
        end_idx = int(len(trim_time) * 0.97)

        time_smooth = trim_time[start_idx:end_idx]
        acc_grad_to_time_smooth = acc_grad_to_time[start_idx:end_idx]

        # save the results
        np.savez_compressed(recency_rerank_file, 
                            trim_acc_ls=trim_acc_ls,
                            mean_acc=mean_acc,
                            trim_time=trim_time,
                            acc_grad_to_time=acc_grad_to_time,
                            time_smooth=time_smooth,
                            acc_grad_to_time_smooth=acc_grad_to_time_smooth)
        
        np.savez_compressed(os.path.join(CUR_DIR, '../data/recency_rerank.npz'), time_smooth=time_smooth, acc_grad_to_time_smooth=acc_grad_to_time_smooth)


if __name__ == "__main__":
    main()