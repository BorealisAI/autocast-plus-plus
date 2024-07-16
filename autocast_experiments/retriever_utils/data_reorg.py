# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import json
import argparse
import datetime
import pandas as pd
import numpy as np
import pdb
from tqdm import tqdm
CUR_DIR = os.path.abspath(os.path.dirname(__file__))
PROJ_DIR = os.path.abspath(os.path.join(CUR_DIR, "../.."))


def arg_parser():
    """
    Parse the command line arguments.
    """
    args = argparse.ArgumentParser()

    # option for the input data
    args.add_argument('--static_data', type=str, required=True, help='path of json data')
    args.add_argument('--rel_csv', type=str, required=True, help='path of relevance csv')
    args.add_argument('--sum_csv', type=str, required=True, help='path of summary csv')
    args.add_argument('--recency_npz', type=str, required=True, help='path of recency npz')

    return args.parse_args()


def load_json_data(data_path):
    """
    Load the json question data from the given path
    """
    assert data_path.endswith('.json')
    with open(data_path, 'r') as fin:
        data = json.load(fin)

    examples = []
    for k, example in enumerate(data):
        example['question'] = example['question'].strip()
        examples.append(example)

    return examples


def main():
    """
    init
    """
    args = arg_parser()

    # load data
    static_data = load_json_data(args.static_data)
    relevance_data = pd.read_csv(args.rel_csv)
    summary_data = pd.read_csv(args.sum_csv)
    recency_data = np.load(args.recency_npz)
    recency_time, recency_score = recency_data['time_smooth'], recency_data['acc_grad_to_time_smooth']

    # process news attributes
    for data_point in tqdm(static_data, desc="Processing data..."):
        # skip the question if there is no news
        if len(data_point['ctxs']) == 0:
            continue

        # read the question and its attributes
        this_q_id = data_point['question_id']
        this_q_body = data_point['question']
        this_start_date = datetime.datetime.strptime(data_point['question_start'][:19], '%Y-%m-%d %H:%M:%S')
        this_expiry_date = datetime.datetime.strptime(data_point['question_expiry'][:19], '%Y-%m-%d %H:%M:%S')
        this_total_minutes = (this_expiry_date - this_start_date).total_seconds() / 60

        if this_q_body.endswith('\r\n'):
            this_q_body = this_q_body[:-2] + '\n'

        flag_q_matching = (relevance_data['question_id'] == this_q_id) & (relevance_data['question'] == this_q_body)
        assert flag_q_matching.sum(), 'No matching question found in relevance data for question ID {}, and question body {}'.format(this_q_id, this_q_body)

        for i_news, news in enumerate(data_point['ctxs']):
            # update the relevance score
            gpt_rel = relevance_data[flag_q_matching & (relevance_data['ctxs_id'] == int(news['id']))]['chatgpt_response_parsed']

            if gpt_rel.shape[0] != 1:
                print('No matching news found in relevance data for question ID {}, question body {}, news id {}'.format(this_q_id, this_q_body, news['id']))
                pdb.set_trace()
            assert gpt_rel.shape[0] == 1, 'More than one matching news found in relevance data for question ID {}, question body {}, news id {}'.format(this_q_id, this_q_body, news['id'])
            gpt_rel_num = eval(gpt_rel.iloc[0])
            gpt_rel_num = [float(item) for item in gpt_rel_num]
            data_point['ctxs'][i_news]['relevance'] = gpt_rel_num

            # update the text summary
            gpt_sum = summary_data[summary_data['ctxs_id'] == int(news['id'])]['chatgpt_response_parsed']
            assert gpt_sum.shape[0] == 1, 'More than one matching news found in summary data for question ID {}, and question body {}, news id {}'.format(this_q_id, this_q_body, news['id'])
            gpt_sum_text = gpt_sum.iloc[0]
            data_point['ctxs'][i_news]['text'] = gpt_sum_text

            # update normalized time
            news_date = datetime.datetime.strptime(news['date'], '%Y-%m-%d')
            minutes_after_publish = (news_date - this_start_date).total_seconds() / 60
            normalized_time = np.clip(minutes_after_publish / this_total_minutes, 0, 1)
            data_point['ctxs'][i_news]['normalized_time'] = normalized_time

            # update recency score
            this_recency = np.interp(normalized_time, recency_time, recency_score)  # bound by the values in recency_score
            data_point['ctxs'][i_news]['recency'] = this_recency

    # save the updated data
    out_path = args.static_data.replace('.json', '_reorg.json')
    with open(out_path, "w", encoding='utf-8') as writer:
        writer.write(json.dumps(static_data, indent=4, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
