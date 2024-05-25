# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the Autocast (https://arxiv.org/abs/2206.15474) implementation
# from https://github.com/andyzoujm/autocast by Andy Zou and Tristan Xiao and Ryan Jia and Joe Kwon and Mantas Mazeika and Richard Li and Dawn Song and Jacob Steinhardt and Owain Evans and Dan Hendrycks
####################################################################################


import json
import time
import datetime
from datetime import timedelta
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank

import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def save_results(
    questions,
    question_answers,
    question_choices,
    question_targets,
    question_ids,
    question_expiries,
    out_file,
):
    merged_data = []
    for i, q in enumerate(questions):
        q_id = question_ids[i]
        q_answers = question_answers[i]
        q_choices = question_choices[i]
        q_targets = question_targets[i]
        expiry = question_expiries[i]

        merged_data.append(
            {
                "question_id": q_id,
                "question": q,
                "answers": str(q_answers),
                "choices": q_choices,
                "question_expiry": expiry,
                "targets": [
                    {
                        "date": index,
                        "target": (
                            str(row["target"])
                            if "target" in row
                            else [
                                str(val)
                                for val in row.values.tolist()
                                if str(val).replace(".", "", 1).isdigit()
                            ]
                        ),
                        "ctxs": row["ctxs"],
                    }
                    for index, row in q_targets.iterrows()
                ],
                "field": None,
            }
        )

    with open(out_file, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(merged_data, indent=4, ensure_ascii=False) + "\n")
    print("Saved results * scores  to %s", out_file)


def main():
    parser = argparse.ArgumentParser(description="Arguments for BM25+CE retriever.")
    parser.add_argument(
        "--beginning", type=str, required=True, help="startg retrieving on this date"
    )
    parser.add_argument(
        "--expiry", type=str, required=True, help="finish retrieving on this date"
    )
    parser.add_argument(
        "--n_docs",
        type=int,
        required=True,
        help="retrieve n daily articles for each question",
    )
    parser.add_argument("--out_file", type=str, required=True, help="output file")
    cfg = parser.parse_args()

    # get questions & answers
    questions = []
    question_choices = []
    question_answers = []
    question_targets = []
    question_ids = []
    question_expiries = []

    ds_key = "autocast"

    assert cfg.beginning and cfg.expiry
    start_date = datetime.datetime.strptime(cfg.beginning, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(cfg.expiry, "%Y-%m-%d")

    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    dates = [str(date.date()) for date in daterange(start_date, end_date)]
    date_to_question_idx = [[] for _ in dates]

    autocast_questions = json.load(open("autocast_questions.json"))
    autocast_questions = [q for q in autocast_questions if q["status"] == "Resolved"]
    time0 = time.time()
    for question_idx, ds_item in tqdm(
        enumerate(autocast_questions), desc="Reading questions..."
    ):
        question = ds_item["question"]
        background = ds_item["background"]
        answers = [ds_item["answer"]]
        choices = ds_item["choices"]
        qid = ds_item["id"]
        expiry = ds_item["close_time"]

        if ds_item["qtype"] != "mc":
            df = pd.DataFrame(ds_item["crowd"])
            df["date"] = df["timestamp"].map(lambda x: x[:10])
            crowd = (
                df.groupby("date")
                .mean(numeric_only=True)
                .rename(columns={df.columns[1]: "target"})
            )
            crowd_preds = crowd
        else:
            df = pd.DataFrame(ds_item["crowd"])
            df["date"] = df["timestamp"].map(lambda x: x[:10])
            fs = np.array(df["forecast"].values.tolist())
            for i in range(fs.shape[1]):
                df[f"{i}"] = fs[:, i]
            crowd = df.groupby("date").mean(numeric_only=True)
            crowd_preds = crowd

        crowd_preds.drop(crowd_preds.tail(1).index, inplace=True)  # avoid leakage
        crowd_preds["ctxs"] = None
        questions.append(question)
        question_choices.append(choices)
        question_answers.append(answers)
        question_targets.append(crowd_preds)
        question_ids.append(qid)
        question_expiries.append(expiry)

        for date_idx, date in enumerate(dates):
            if date in crowd_preds.index:
                date_to_question_idx[date_idx].append(question_idx)
    print("Reading questions took {:f} sec.".format(time.time() - time0))

    # for BM25, we need to re-initialize the elasticsearch-based utility for each iteration in the for loop to avoid error
    # we run it here just to ensure the elasticsearch is running
    model = BM25(
        hostname="http://localhost:9200",
        index_name=ds_key + "_rainbowquartz_bm25_ce",
        initialize=True,
        sleep_for=1.0,
    )
    retriever = EvaluateRetrieval(model)

    # for CE, we just need to initialize for once (it's a neural network essentially)
    cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-electra-base")
    reranker = Rerank(cross_encoder_model, batch_size=256)

    from datasets import Dataset
    from datasets.utils.logging import set_verbosity_error

    set_verbosity_error()

    cc_news_dataset = Dataset.load_from_disk(
        os.path.join(PROJECT_DIR, "datasets/cc_news")
    )

    # create index here once and for all
    cc_news_dataset = cc_news_dataset.add_column("id", range(len(cc_news_dataset)))

    flag_mini_batch = False  # use dataset generator for cc_news_dataset to avoid huge RAM use [slow, don't use!]
    flag_split_by_yr_mn = False  # split cc_news_dataset by year and month to avoid huge RAM use [still slow...]
    assert not (
        flag_split_by_yr_mn and flag_mini_batch
    ), "Cannot use both flags at the same time!"
    if flag_mini_batch:
        raise ValueError(
            "Mini-batched dataloder is still too slow! The code is kept for the sake of completeness only. Do not use!"
        )
        dataset_bs = 10240
        cc_news_df = cc_news_dataset.to_pandas(
            batched=True, batch_size=dataset_bs
        )  # mini-batch mode
    elif flag_split_by_yr_mn:
        # split the dataset by year to speed up searching by date later
        yr_beginning, yr_expiry = int(cfg.beginning[:4]), int(cfg.expiry[:4])
        print(
            "Splitting CC News dataset by year from {:d} to {:d}.".format(
                yr_beginning, yr_expiry
            )
        )
        cc_news_dataset_per_yr = {}
        for yr in range(yr_beginning, yr_expiry + 1):
            time0 = time.time()
            print("Starting to split CC News dataset by year {:d}...".format(yr))
            cc_news_dataset_per_mn = {}
            cc_news_dataset_this_year = cc_news_dataset.filter(
                lambda x: x[:4] == str(yr),
                input_columns=["date"],
                num_proc=os.cpu_count(),
            )
            for mn in range(1, 13):
                cc_news_dataset_per_mn[mn] = cc_news_dataset_this_year.filter(
                    lambda x: x[5:7] == "{:02d}".format(mn),
                    input_columns=["date"],
                    num_proc=os.cpu_count(),
                )
            cc_news_dataset_per_yr[yr] = cc_news_dataset_per_mn
            print(
                "Finished splitting CC News dataset by year {:d}. Time: {:f} sec.".format(
                    yr, time.time() - time0
                )
            )
            for mn in range(1, 13):
                print(
                    "Breakdown: Year {:d} Month {:02d}: {:d} articles".format(
                        yr, mn, len(cc_news_dataset_per_yr[yr][mn])
                    )
                )

        # sanity check on year
        for date in dates:
            yr = int(date[:4])
            assert (
                yr in cc_news_dataset_per_yr
            ), "Year {:d} not found in cc_news_dataset_per_yr!".format(yr)
    else:
        cc_news_df = cc_news_dataset.to_pandas()  # load all data in memory

    _beginning_date, _ending_date = (
        cc_news_dataset["date"][0],
        cc_news_dataset["date"][-1],
    )
    print(
        "CC News dataset loaded. Total number of articles: {:d} spanning from {:s} to {:s}.".format(
            len(cc_news_dataset), _beginning_date, _ending_date
        )
    )
    print("Start retrieving articles in the for loop...")

    for date_idx, date in tqdm(enumerate(dates), desc="Retrieving articles..."):
        # obtain news articles for current date
        if flag_mini_batch:
            raise ValueError(
                "Mini-batched dataloder is still too slow! The code is kept for the sake of completeness only. Do not use!"
            )
            cc_news_df_daily, cc_news_df_idx = [], []
            for i, cc_news_df_batch in enumerate(cc_news_df):
                cc_news_df_daily_batch = cc_news_df_batch[
                    cc_news_df_batch["date"] == date
                ]
                batch_idx = cc_news_df_batch[
                    cc_news_df_batch["date"] == date
                ].index.tolist()
                batch_idx = [
                    idx + i * dataset_bs for idx in batch_idx
                ]  # offset the batch idx by the previous batch
                if len(cc_news_df_daily_batch):
                    if len(cc_news_df_daily) == 0:
                        cc_news_df_daily = (
                            cc_news_df_daily_batch  # list -> pandas dataframe
                        )
                    else:
                        cc_news_df_daily = cc_news_df_daily.append(
                            cc_news_df_daily_batch
                        )  # dataframe append
                    cc_news_df_idx.extend(batch_idx)
            # reset iterator for each run
            cc_news_df = cc_news_dataset.to_pandas(batched=True, batch_size=dataset_bs)
            if len(cc_news_df_daily) == 0:
                continue
            # recover the index
            cc_news_df_daily = cc_news_df_daily.assign(id=cc_news_df_idx)
        elif flag_split_by_yr_mn:
            cc_news_ds_daily = cc_news_dataset_per_yr[int(date[:4])][
                int(date[5:7])
            ].filter(
                lambda x: x == date, input_columns=["date"]
            )  # dataset
            if len(cc_news_ds_daily) == 0:
                continue
            cc_news_df_daily = cc_news_ds_daily.to_pandas()  # dataset -> dataframe
        else:
            cc_news_df_daily = cc_news_df[cc_news_df["date"] == date]
            if len(cc_news_df_daily) == 0:
                continue

        print(
            "Retrive non-empty articles progress: {:d}/{:d}, date: {:s}".format(
                date_idx + 1, len(dates), date
            )
        )

        # process the daily news articles
        k = min(cfg.n_docs, len(cc_news_df_daily))

        ids = cc_news_df_daily["id"].values.tolist()
        titles = cc_news_df_daily["title"].values.tolist()
        texts = cc_news_df_daily["text"].values.tolist()

        daily_corpus = {}
        for i in range(len(ids)):
            json_obj = {}
            json_obj["title"] = titles[i]
            json_obj["text"] = texts[i]
            daily_corpus[str(ids[i])] = json_obj

        # obtain query questions at this date
        question_indices = date_to_question_idx[date_idx]
        daily_queries = {}
        for question_idx in question_indices:
            question = questions[question_idx]
            id = str(ds_key) + "_" + str(question_idx)
            daily_queries[id] = question

        if len(daily_queries) == 0:
            print("no queries for: " + str(date))
            continue

        # start retrieval, note the elasticsearch service must be running
        retriever.retriever.initialise()
        retriever.retriever.results = {}
        try:
            scores = retriever.retrieve(daily_corpus, daily_queries)
            print("retrieval done")
        except Exception as e:
            print("retrieval exception: " + str(e))
            continue

        # cross-encoder ranking, note the GPU can be used here to speed up
        try:
            rerank_scores = reranker.rerank(
                daily_corpus, daily_queries, scores, top_k=min(100, k)
            )
            print("reranking done")
        except Exception as e:
            print("reranking exception: " + str(e))
            continue

        for score_idx in rerank_scores:
            top_k = list(rerank_scores[score_idx].items())[:k]

            ctxs = [
                {
                    "id": doc_id,
                    "title": daily_corpus[doc_id]["title"],
                    "text": daily_corpus[doc_id]["text"],
                    "score": score,
                }
                for doc_id, score in top_k
            ]

            question_idx = int(score_idx.split("_")[-1])
            question_targets[question_idx].at[date, "ctxs"] = ctxs

        if date_idx % 100 == 0:
            print(f"\n{'='*20}\nDone retrieval for 100 days, now at {date}\n{'='*20}\n")
            print("time: %f sec.", time.time() - time0)
            time0 = time.time()

    save_results(
        questions,
        question_answers,
        question_choices,
        question_targets,
        question_ids,
        question_expiries,
        cfg.out_file,
    )


if __name__ == "__main__":
    main()
