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


import torch
import json
import copy
import numpy as np
from copy import deepcopy


class Dataset(torch.utils.data.Dataset):
    """
    Dataset for the QA data.
    """

    def __init__(
        self,
        data,
        logger,
        n_context=None,
        question_prefix="question:",
        title_prefix="title:",
        passage_prefix="context:",
        choices_prefix="choices:",
        bound_prefix="bounds:",
        score_threshold=0.0,
        mode="train",
        train_with_news=True,
        numerical_bins=20,
    ):
        self.data = data
        self.logger = logger
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.choices_prefix = choices_prefix
        self.bound_prefix = bound_prefix
        self.score_threshold = score_threshold
        self.mode = mode
        self.train_with_news = train_with_news
        self.numerical_bins = numerical_bins

        self.pre_filter()

    def pre_filter(self):
        self.data_by_class = {}
        valid_data = []
        for example in self.data:
            label = example["qtype"]
            if label == "num":
                # bin the numerical answer
                example["answers_bin_num"] = (
                    np.digitize(
                        example["answers"], bins=np.linspace(0, 1, self.numerical_bins)
                    )
                    - 1
                )
                example["answers_bin_char"] = chr(example["answers_bin_num"] + ord("A"))

            if self.score_threshold:
                # when score threshold is applied
                # only consider 1) questions with news articles
                # only use 2) articles with retrieval score > threshold
                if len(example["ctxs"]) == 0:
                    continue
                else:
                    ctxs_original = copy.deepcopy(example["ctxs"])
                    for ctx in ctxs_original:
                        if ctx["score"] < self.score_threshold:
                            example["ctxs"].remove(ctx)
                    if len(example["ctxs"]) == 0:
                        continue
                assert len(example["ctxs"]) > 0, "No ctxs with score > {}".format(
                    self.score_threshold
                )

            if self.mode == "train" and self.train_with_news:
                # when training with news articles is a must
                if len(example["ctxs"]) == 0:
                    continue

            # aggregate the relevance and recency scores
            if len(example["ctxs"]) > 0:
                scores_relevance_raw = np.stack(
                    [c["relevance"] for c in example["ctxs"]], axis=0
                ).clip(0, 4)
                scores_recency_raw = np.array([c["recency"] for c in example["ctxs"]])

                # allowed relevance score is 0, 1, 2, 3, 4
                assert set(np.unique(scores_relevance_raw)) <= set([0, 1, 2, 3, 4])
                assert (
                    scores_relevance_raw.min() >= 0 and scores_relevance_raw.max() <= 4
                )
                assert scores_recency_raw.min() >= 0 and scores_recency_raw.max() <= 1

                scores_relevance = np.clip(
                    scores_relevance_raw.mean(axis=-1) / 4.0, a_min=1e-4, a_max=1.0
                )
                scores_final = scores_relevance * scores_recency_raw

                # add the scores to the example
                for i, ctx in enumerate(example["ctxs"]):
                    ctx["score_final"] = scores_final[i]

                # sort the ctxs by the final scores from high to low
                example["ctxs"] = sorted(
                    example["ctxs"], key=lambda x: x["score_final"], reverse=True
                )

            if label not in self.data_by_class:
                self.data_by_class[label] = []
            self.data_by_class[label].append(example)
            valid_data.append(example)
        self.data = valid_data

        output_str = ""
        for label in self.data_by_class:
            output_str += f"{len(self.data_by_class[label])} {label} "
        self.logger.info(
            "{:s} dataloader: #samples by class: {:s}".format(self.mode, output_str)
        )

    def get_target(self, example):
        if isinstance(example["choices"], dict):
            # numerical prediction
            assert example["qtype"] == "num"
            assert isinstance(example["answers"], float)
            assert isinstance(example["answers_bin_char"], str)
            return example["answers"], example["answers_bin_char"]
        elif isinstance(example["choices"], list):
            # t/f or mc
            assert example["qtype"] in ["t/f", "mc"]
            assert isinstance(example["answers"], str)
            return example["answers"], example["answers"]
        else:
            raise NotImplementedError

    def get_example(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example["question"]
        question += ", start time: " + example["question_start"][:10]
        question += ", expiry time: " + example["question_expiry"][:10]
        choices = example["choices"]
        target_raw = str(self.get_target(example)[0])
        target = str(self.get_target(example)[1])

        if example["qtype"] in ["t/f", "mc"]:
            # Append available choices for t/f and MC questions
            assert (
                not target[:-5].lower().strip() in ["yes", "no"]
            ) and not isinstance(choices, dict)
            choices = [
                chr(i + ord("A")) + ": " + choices[i] for i in range(len(choices))
            ]
            question = (
                question + ", " + self.choices_prefix + " " + " | ".join(choices) + "."
            )
        elif example["qtype"] == "num":
            # Append for numerical questions
            assert isinstance(choices, dict)
            min, max, deriv = (
                str(choices["min"]),
                str(choices["max"]),
                str(choices["deriv_ratio"]),
            )
            num_letters = [chr(i + ord("A")) for i in range(self.numerical_bins)]
            question = (
                question
                + ", "
                + self.bound_prefix
                + " min: "
                + min
                + " | max: "
                + max
                + " | deriv: "
                + deriv
            )
            question += (
                ", "
                + self.choices_prefix
                + f" {' | '.join(num_letters)}, with {self.numerical_bins} bins evenly distributed in range of 0 to 1."
            )
        else:
            raise NotImplementedError

        # static news articles
        if "ctxs" in example and len(example["ctxs"]) and self.n_context is not None:
            f = (
                "news date:"
                + " {}, "
                + self.title_prefix
                + " {}, question-news content relevance: {}, "
                + self.passage_prefix
                + " {}."
            )
            if (
                len(example["ctxs"]) < self.n_context
            ):  # if we don't have enough articles
                add_on = self.n_context - len(example["ctxs"])
                example["ctxs"].extend(
                    [example["ctxs"][0]] * add_on
                )  # duplicate the first (most related) article
            contexts = example["ctxs"][: self.n_context]

            def _relevance_to_txt(in_rel_score):
                assert in_rel_score >= 0 and in_rel_score <= 1
                if in_rel_score < 0.1:
                    return "very low"
                elif in_rel_score < 0.3:
                    return "low"
                elif in_rel_score < 0.5:
                    return "medium"
                elif in_rel_score < 0.7:
                    return "high"
                else:
                    return "very high"

            txt_relevance = [
                _relevance_to_txt(np.mean(c["relevance"]) / 4.0) for c in contexts
            ]
            passages = [
                f.format(c["date"], c["title"], txt_relevance[i], c["text"])
                for i, c in enumerate(contexts)
            ]

            # type: T/F, higher human forecast means more likely to be true
            # type: MC, higher human forecast means which answer is more likely to be correct
            # type: NUM, higher human forecast means which bin is more likely to be correct
            if example["qtype"] == "t/f":
                if target == "yes":
                    human_forecast = [c["forecast"] for i, c in enumerate(contexts)]
                else:
                    human_forecast = [
                        1.0 - c["forecast"] for i, c in enumerate(contexts)
                    ]
            elif example["qtype"] == "mc":
                answer_idx = ord(target) - ord("A")
                human_forecast = [
                    c["forecast"][answer_idx] for i, c in enumerate(contexts)
                ]
            elif example["qtype"] == "num":
                human_forecast = [c["forecast"] for i, c in enumerate(contexts)]

            # get the forecast temporal order
            normalized_time_forecast = [c["normalized_time"] for c in contexts]
            forecast_time_order = np.argsort(normalized_time_forecast)
        else:
            passages = None
            human_forecast = None
            forecast_time_order = None

        return {
            "index": index,
            "id": example["question_id"],
            "question": question,
            "end_time": example["question_expiry"],
            "target_raw": target_raw,
            "target": target,
            "choices": choices,
            "passages": passages,
            "human_forecast": human_forecast,
            "forecast_time_order": forecast_time_order,
        }


def encode_passages(batch_text_passages, tokenizer, max_length):
    """
    Encode the text passages into tokenized ids and masks
    """
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        text_passages = [tp for tp in text_passages]
        p = tokenizer.batch_encode_plus(
            text_passages,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            truncation=True,
        )
        passage_ids.append(p["input_ids"][None])
        passage_masks.append(p["attention_mask"][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()


class Collator(object):
    """
    Collator for the QA data.
    """

    def __init__(self, text_maxlength, tokenizer, answer_maxlength=10, n_context=0):
        self.text_maxlength = text_maxlength
        self.tokenizer = tokenizer
        self.answer_maxlength = answer_maxlength
        self.n_context = n_context

    def __call__(self, batch):
        index = torch.tensor(
            [ex["index"] for ex in batch]
        )  # numerical index within the dataset
        ids = [ex["id"] for ex in batch]  # question id, e.g., G7
        targets = [
            ex["target"] for ex in batch
        ]  # question answers (for training, binned numerical answers)
        targets_raw = [
            ex["target_raw"] for ex in batch
        ]  # question answers (raw data, unbinned numerical answers)
        choices = [ex["choices"] for ex in batch]  # question choices
        human_forecast = [
            ex["human_forecast"] for ex in batch
        ]  # human forecast for the news articles
        forecast_time_order = [
            ex["forecast_time_order"] for ex in batch
        ]  # temporal order of the news articles

        # collect the indices for questions of different type
        # t/f, mc, numerical
        tf_indices, mc_indices, num_indices = [], [], []
        for i in range(len(targets)):
            if not isinstance(choices[i], dict):
                # t/f or mc type
                if targets[i].lower().strip() in ["yes", "no"]:
                    tf_indices.append(i)
                else:
                    mc_indices.append(i)
            else:
                num_indices.append(i)

        # count the frequency of each question type and collect their indices
        tf_len, mc_len, num_len = len(tf_indices), len(mc_indices), len(num_indices)
        length = max(tf_len, mc_len, num_len)
        if tf_len < length:
            tf_indices = tf_indices + [-1] * (length - tf_len)
        if mc_len < length:
            mc_indices = mc_indices + [-1] * (length - mc_len)
        if num_len < length:
            num_indices = num_indices + [-1] * (length - num_len)
        lengths = torch.tensor([tf_len, mc_len, num_len])  # [3]
        indices = torch.tensor([tf_indices] + [mc_indices] + [num_indices])  # [3, N]

        # pass the targets (question answers) into tokenizers
        targets_tokenized = self.tokenizer.batch_encode_plus(
            targets,
            padding="max_length",
            max_length=self.answer_maxlength,
            return_tensors="pt",
            truncation=True,
        )
        targets_ids = targets_tokenized["input_ids"]
        targets_mask = targets_tokenized["attention_mask"].bool()
        targets_ids = targets_ids.masked_fill(
            ~targets_mask, -100
        )  # -100 is the padding id

        # collect the labels for questions of different type for conditional generation
        labels = []
        for i in range(len(index)):
            if not isinstance(choices[i], dict):
                # t/f or mc type
                labels.append(targets_ids[i])
            else:
                # numerical type
                labels.append(
                    targets_ids[i]
                )  # same as t/f or mc type for binned numerical answers
        labels = torch.stack(labels).to(torch.float32)

        # prepend the question prefix to the passage
        def append_question(example, n):
            # prepend the question to the passage and return merged text and article scores
            if example["passages"] is None:
                if n is not None:
                    return [
                        example["question"]
                    ] * n  # repeat the question n times if there is no news articles at all
                return [example["question"]]
            return [example["question"] + " " + t for t in example["passages"]]

        text_passages = []
        for example in batch:
            text_ = append_question(example, self.n_context)
            text_passages.append(text_)
        passage_ids, passage_masks = encode_passages(
            text_passages, self.tokenizer, self.text_maxlength
        )  # tokenized ids and masks

        human_forecasts = []
        forecast_time_orders = []
        for forecast, time_order in zip(human_forecast, forecast_time_order):
            if forecast is None:
                human_forecasts.append(torch.full((self.n_context,), -100.0))
                forecast_time_orders.append(torch.full((self.n_context,), -100))
            else:
                human_forecasts.append(torch.tensor(forecast))
                forecast_time_orders.append(torch.tensor(time_order))

        human_forecasts = torch.stack(human_forecasts)  # [B, N]
        forecast_time_orders = torch.stack(forecast_time_orders)  # [B, N]

        # returned parameters
        # index: dataset-specific integer index, tensor of integers                         [B]
        # ids: original question IDs, list of strings                                       [B]
        # labels: tokenized question answers, tensor of integers                            [B, L=10]
        # indices: question types, tensor of integers, t/f + mc + num                       [3, N]
        # lengths: count of questions of each type, tensor of integers, t/f + mc + num      [3]
        # passage_ids: tokenized passage IDs, tensor of integers                            [B, N, D=512]
        # passage_masks: passage masks, tensor of booleans                                  [B, N, D=512]
        # targets_raw: raw question answers, list of strings                                [B]
        # human_forecasts: human forecasts for the news articles, tensor of floats          [B, N]
        # forecast_time_orders: temporal order of the news articles, tensor of integers     [B, N]
        return (
            index,
            ids,
            labels,
            indices,
            lengths,
            passage_ids,
            passage_masks,
            targets_raw,
            human_forecasts,
            forecast_time_orders,
        )


def load_json_data(data_path):
    """
    Load json data from a file
    """
    with open(data_path, "r") as fin:
        data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        examples.append(example)

    return examples


def load_datasets(opt, logger):
    """
    Load data for training and evaluation
    """
    train_examples = load_json_data(opt.train_data)
    if opt.debug:
        _n_datapoints = 50
        _n_tf, _n_mc = int(_n_datapoints / 3), int(_n_datapoints / 3)
        _n_num = _n_datapoints - _n_tf - _n_mc
        out_train_examples = []
        for ex in train_examples:
            if len(ex["ctxs"]):
                if ex["qtype"] == "t/f" and _n_tf > 0:
                    out_train_examples.append(ex)
                    _n_tf -= 1
                elif ex["qtype"] == "mc" and _n_mc > 0:
                    out_train_examples.append(ex)
                    _n_mc -= 1
                elif ex["qtype"] == "num" and _n_num > 0:
                    out_train_examples.append(ex)
                    _n_num -= 1
            if _n_tf == 0 and _n_mc == 0 and _n_num == 0:
                break
        train_examples = out_train_examples  # select a small subset
        assert (
            len(out_train_examples) == _n_datapoints
        ), "DEBUG mode: not enough examples"
    train_dataset = Dataset(
        train_examples,
        logger,
        opt.n_context,
        score_threshold=opt.score_threshold,
        mode="train",
        train_with_news=opt.train_with_news,
    )
    opt.train_data_size = len(
        train_dataset
    )  # overwrite to get the actual training data size

    if opt.debug:
        logger.info(f"DEBUG mode: overfitting on training examples.")
        eval_dataset = deepcopy(train_dataset)  # overfitting on training examples
    else:
        eval_examples = load_json_data(opt.eval_data)
        eval_dataset = Dataset(eval_examples, logger, opt.n_context, mode="eval")

    return train_dataset, eval_dataset
