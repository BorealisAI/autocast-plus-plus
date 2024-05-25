#!/bin/sh

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


export B='2016-01-01'
export E='2022-04-12'
export N_DOCS=10

python retrieve_cc_news_bm25+ce.py \
  --out_file=autocast_cc_news_retrieved_top_${N_DOCS}.json \
  --n_docs=$N_DOCS \
  --beginning=$B \
  --expiry=$E
