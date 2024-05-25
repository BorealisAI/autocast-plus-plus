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


import argparse
import json
import copy
import os
import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CUR_DIR = os.path.abspath(os.path.dirname(__file__))


def parse_arguments():
    """
    Parse the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", help="Whether to run in debug mode."
    )
    parser.add_argument(
        "--n_doc", type=int, default=30, help="Number of documents to retrieve."
    )
    parser.add_argument(
        "--static_only", action="store_true", help="Whether to only save static data."
    )
    return parser.parse_args()


def load_json_data(flag_debug_mode):
    """
    Load the raw retrieved data and create the negated questions.
    """
    # load retrieved data
    if flag_debug_mode:
        retrieved_data = os.path.join(CUR_DIR, "autocast_cc_news_retrieved_debug.json")
    else:
        retrieved_data = os.path.join(CUR_DIR, "autocast_cc_news_retrieved.json")
    retrieved_data = os.path.join(os.path.dirname(__file__), retrieved_data)
    print(f"Start loading data from {retrieved_data}")
    data = json.load(open(retrieved_data))

    # load questions and created negated ones
    negated_questions = json.load(
        open(os.path.join(CUR_DIR, "negated_tf_questions.json"))
    )
    autocast_questions = json.load(
        open(os.path.join(CUR_DIR, "autocast_questions.json"))
    )
    qid_to_question = {q["id"]: q for q in autocast_questions}
    qid_to_negation = {q["id"]: q for q in negated_questions}
    all_questions = []
    for idx, q in enumerate(data):
        q["qtype"] = qid_to_question[q["question_id"]]["qtype"]
        q["question_start"] = qid_to_question[q["question_id"]]["publish_time"]

        # fix string conversion issue for question answers
        q["answers"] = eval(q["answers"])[0]
        if q["qtype"] == "t/f":
            assert q["answers"] in ["yes", "no"]
        elif q["qtype"] == "mc":
            assert q["answers"] in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K"]
        elif q["qtype"] == "num":
            assert isinstance(q["answers"], float)
            if not (q["answers"] >= 0 and q["answers"] <= 1) and flag_debug_mode:
                print(
                    "Num answer exception found! ID: {}, q['answers'] is: {}, will be clipped to [0, 1]. ".format(
                        q["question_id"], q["answers"]
                    )
                )
            q["answers"] = np.clip(q["answers"], 0.0, 1.0)
        else:
            raise ValueError(f'Unknown question type: {q["qtype"]}')

        # fix string conversion issue for target probabilities (crowd forecasts)
        for day in q["targets"]:
            if isinstance(day["target"], str):
                day["target"] = eval(day["target"])
                assert isinstance(day["target"], float)
                if not (day["target"] >= 0 and day["target"] <= 1) and flag_debug_mode:
                    print(
                        "Crowd target exception found! ID: {}, date: {}, target is: {}, will be clipped to [0, 1].".format(
                            q["question_id"], day["date"], day["target"]
                        )
                    )
                day["target"] = np.clip(day["target"], 0.0, 1.0)
            elif isinstance(day["target"], list):
                day["target"] = [eval(t) for t in day["target"]]
                assert all([isinstance(t, float) for t in day["target"]])
                for t in day["target"]:
                    if not (t >= 0 and t <= 1) and flag_debug_mode:
                        print(
                            "Crowd target exception found! ID: {}, date: {}, target is: {}, will be clipped to [0, 1].".format(
                                q["question_id"], day["date"], day["target"]
                            )
                        )
                day["target"] = [np.clip(t, 0.0, 1.0) for t in day["target"]]

        # create negated questions
        if q["question_id"] in qid_to_negation:
            negated_q = copy.deepcopy(q)
            negated_q["question"] = qid_to_negation[q["question_id"]]["negated"]
            for day in negated_q["targets"]:
                day["target"] = 1 - float(
                    day["target"]
                )  # flip the forecast probabilities
            if q["answers"] == "yes":
                negated_q["answers"] = "no"
            else:
                negated_q["answers"] = "yes"
            all_questions.append(negated_q)

        all_questions.append(q)

    # print out the first 3 questions of each type
    print(
        "autocast questions loading is finished. Preview of the first three questions of each type:"
    )
    counter_displayed = {"t/f": 0, "mc": 0, "num": 0}
    max_displayed = {"t/f": 3, "mc": 3, "num": 3}
    for idx, q in enumerate(data):
        flag_display = False
        if q["qtype"] == "t/f":
            if counter_displayed["t/f"] < max_displayed["t/f"]:
                flag_display = True
            counter_displayed["t/f"] += 1
        elif q["qtype"] == "mc":
            if counter_displayed["mc"] < max_displayed["mc"]:
                flag_display = True
            counter_displayed["mc"] += 1
        elif q["qtype"] == "num":
            if counter_displayed["num"] < max_displayed["num"]:
                flag_display = True
            counter_displayed["num"] += 1
        else:
            raise ValueError("Unknown question type: {}".format(q["qtype"]))
        if flag_display:
            print(
                "Idx: {}, Answer: {}, Answer type: {}, Type: {}".format(
                    idx, q["answers"], type(q["answers"]), q["qtype"]
                )
            )

    return all_questions


def save_temporal_data(all_questions, flag_debug_mode, saving_name_suffix):
    """
    Save temporal data.
    """
    print("Start to do temporal spliting...")
    c1, c2 = 0, 0
    temp_train_qs, temp_test_qs = [], []
    for d in all_questions:
        if d["question_expiry"] < "2021-05-11":
            c1 += 1
            temp_train_qs.append(d)
        else:
            temp_test_qs.append(d)
            c2 += 1
    print("Temporal split: train: {}, test: {}".format(c1, c2))

    with open(
        os.path.join(CUR_DIR, f"temporal_train{saving_name_suffix}.json"),
        "w",
        encoding="utf-8",
    ) as writer:
        writer.write(json.dumps(temp_train_qs, indent=4, ensure_ascii=False) + "\n")
    with open(
        os.path.join(CUR_DIR, f"temporal_test{saving_name_suffix}.json"),
        "w",
        encoding="utf-8",
    ) as writer:
        writer.write(json.dumps(temp_test_qs, indent=4, ensure_ascii=False) + "\n")


def save_static_data(all_questions, flag_debug_mode, saving_name_suffix, top_n_ctxs=30):
    """
    Save static data.
    """
    print("Start to do static spliting...")
    method = "bm25ce"
    descending = method != "dpr"

    static_data = []
    for question in all_questions:
        new_question = copy.deepcopy(question)
        del new_question["targets"]

        all_ctxs = []
        all_scores = []
        for target in question["targets"]:
            if target["ctxs"]:
                ctxs_with_date_forecast = copy.deepcopy(target["ctxs"])
                for this_dict in ctxs_with_date_forecast:
                    this_dict["date"] = target["date"]
                    this_dict["forecast"] = target["target"]
                all_ctxs.extend(ctxs_with_date_forecast)
                all_scores.extend([float(ctxs["score"]) for ctxs in target["ctxs"]])

        sorted_idx = [
            x
            for _, x in sorted(
                zip(all_scores, range(len(all_scores))), reverse=descending
            )
        ]
        new_question["ctxs"] = [all_ctxs[i] for i in sorted_idx][
            :top_n_ctxs
        ]  # save the top N contexts

        static_data.append(new_question)

    # split training and testing based on the question expiry date
    c1, c2 = 0, 0
    static_train_qs, static_test_qs = [], []
    for d in static_data:
        if d["question_expiry"] < "2021-05-11":
            c1 += 1
            static_train_qs.append(d)
        else:
            static_test_qs.append(d)
            c2 += 1
    print("Static split: train: {}, test: {}".format(c1, c2))

    with open(
        os.path.join(CUR_DIR, f"static_train{saving_name_suffix}.json"),
        "w",
        encoding="utf-8",
    ) as writer:
        writer.write(json.dumps(static_train_qs, indent=4, ensure_ascii=False) + "\n")
    with open(
        os.path.join(CUR_DIR, f"static_test{saving_name_suffix}.json"),
        "w",
        encoding="utf-8",
    ) as writer:
        writer.write(json.dumps(static_test_qs, indent=4, ensure_ascii=False) + "\n")


def main():
    args = parse_arguments()
    flag_debug_mode = args.debug

    all_questions = load_json_data(flag_debug_mode)

    saving_name_suffix = "_debug" if flag_debug_mode else "_top{:d}".format(args.n_doc)
    save_static_data(
        all_questions, flag_debug_mode, saving_name_suffix, top_n_ctxs=args.n_doc
    )

    if not args.static_only:
        saving_name_suffix = "_debug" if flag_debug_mode else ""
        save_temporal_data(all_questions, flag_debug_mode, saving_name_suffix)


if __name__ == "__main__":
    main()
