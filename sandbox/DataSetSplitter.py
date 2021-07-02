#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import RepeatedKFold
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations
import argparse
import json
import re


def get_ids_dataframe(corpus_path: str) -> pd.DataFrame:
    '''
        in: 
            corpus_path: contains the path with the sub-folders named after the 
                         document types
        out: pandas.DataFrame: [id, group]
    '''

    ids = []
    for subdir, folders, files in os.walk(corpus_path):
        if subdir != corpus_path:
            for file in tqdm(os.listdir(path=subdir)):
                if ".ann" in file:
                    if Path(os.path.join(subdir, file)).stat().st_size != 0:
                        ids.append({
                            "id": file.split(".")[0],
                            "group": re.split(r"[\\\/]", subdir)[-1]
                        }
                        )
    return pd.DataFrame(ids).set_index('id')


def get_intra_group_folds(ids_df: pd.DataFrame,
                          rnd_state: int = 1524513,
                          num_reps: int = 100,
                          num_folds: int = 10) -> dict:
    '''
        in: 
            ids_df: pandas.DataFrame[id, group] 
            rnd_state: seed for random number generator
        
        out: dict {letterype: [(fold_train_0, fold_test_0), (fold_train_1, fold_test1), (..)]}
        
    '''

    intra_group_folds = defaultdict(list)
    groups = ids_df['group'].unique()
    for document_type in groups:
        repeated_k_folder = RepeatedKFold(n_repeats=num_reps, n_splits=num_folds, random_state=rnd_state)
        df = ids_df[ids_df['group'] == document_type]
        for train_indcs, test_indcs in tqdm(repeated_k_folder.split(df)):
            intra_group_folds[document_type].append((list(ids_df.iloc[train_indcs].index),
                                                     list(ids_df.iloc[test_indcs].index)))
    return intra_group_folds


def get_inter_group_splits(ids_df: pd.DataFrame,
                           p: int,
                           rnd_state: int = 1524513,
                           train_frac: int = 1) -> dict:
    '''
        in: 
            ids_df: pandas.DataFrame[id, group]
            p: number of lettertypes in train set
            rnd_state: seed for random number generator
            train_frac: fraction of training data used for model, default=1
        
        out: dict {}
    
    '''

    group_splits = dict()
    unique_groups = ids_df.group.unique()
    groups_train = [tuple(map(str, comb)) for comb in combinations(unique_groups, p)]
    groups_test = [tuple(set(unique_groups) - set(gt)) for gt in groups_train]

    for idx, gtrain in enumerate(groups_train):
        gtest = groups_test[idx]
        group_splits[idx] = {'groups_train': gtrain,
                             'groups_test': gtest,
                             'train_fold': ids_df.loc[ids_df.group.isin(gtrain)] \
                                 .sample(frac=train_frac, random_state=rnd_state).index.tolist(),
                             'test_fold': ids_df.loc[ids_df.group.isin(gtest)].index.tolist()
                             }
    return group_splits


def write_folds(intra: dict, inter: dict, output: str):
    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, "intra_folds.json"), "w") as fp:
        json.dump(intra, fp)

    with open(os.path.join(output, "inter_folds.json"), "w") as fp:
        json.dump(inter, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--corpus_loc', dest='corpus_loc', type=str,
                        help='location of corpus')
    parser.add_argument('--num_folds', dest='num_folds', type=int,
                        help='number of folds, for intra-group', default=10)
    parser.add_argument('--num_reps', dest='num_reps', type=int,
                        help='number of CV repetitions, for intra-group', default=100)
    parser.add_argument('--train_fraction', dest='train_fraction', type=float,
                        help='fraction of training used, for inter-group', default=1.0)
    parser.add_argument('--num_groups', dest='num_groups', type=int,
                        help='number of groups for training, for inter-group', default=3)
    parser.add_argument('--output', dest='output_location', type=str, help="output folder of pickles",
                        default="./output")
    args = parser.parse_args()

    ids_df = get_ids_dataframe(corpus_path=args.corpus_loc)

    print("Generating intra-group folds...")
    intra_folds = get_intra_group_folds(ids_df=ids_df,
                                        num_reps=args.num_reps,
                                        num_folds=args.num_folds
                                        )
    print("Generating inter-group folds...")
    inter_folds = get_inter_group_splits(ids_df=ids_df,
                                         p=args.num_groups,
                                         train_frac=args.train_fraction
                                         )

    print("Writing to disk...")
    write_folds(intra=intra_folds, inter=inter_folds, output=args.output_location)
