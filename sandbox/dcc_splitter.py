#!/usr/bin/env python
# coding: utf-8
from sklearn.model_selection import KFold
from pathlib import Path
import argparse
import json
import numpy as np


DEFAULT_DCC_DIR = Path('../data/EMCDutchClinicalCorpus')
DEFAULT_OUTPUT_DIR = Path('../data')
DEFAULT_SKIP_FILE = Path('../data/DCC_files_to_exclude.json')
DEFAULT_N_SPLITS = 10
DEFAULT_RANDOM_STATE = 1524513


class DCCSplitter:
    def __init__(self, dcc_dir, output_dir, skip_file, n_splits, random_state, write_to_file=False):
        self.dcc_dir = dcc_dir
        self.output_dir = output_dir
        self.skip_file = skip_file
        self.n_splits = n_splits
        self.random_state = random_state
        self.write_to_file = write_to_file

        self.dcc_dir = DEFAULT_DCC_DIR if self.dcc_dir is None else self.dcc_dir
        self.output_dir = DEFAULT_OUTPUT_DIR if self.output_dir is None else self.output_dir
        self.skip_file = DEFAULT_SKIP_FILE if self.skip_file is None else self.skip_file
        self.n_splits = DEFAULT_N_SPLITS if self.n_splits is None else self.n_splits
        self.random_state = DEFAULT_RANDOM_STATE if self.random_state is None else self.random_state

    def split(self):
        dcc_names = self.get_dcc_names(self.dcc_dir, self.skip_file)
        split_list = self.shuffle_split_files(dcc_names, self.n_splits, self.random_state)
        if self.write_to_file:
            self.write_splits(split_list, self.output_dir)
        return split_list

    def get_dcc_names(self, 
                     dcc_dir: Path = DEFAULT_DCC_DIR,
                     skip_file: Path = DEFAULT_SKIP_FILE):

        # Some files should be skipped because their data is corrupted
        with open(skip_file) as json_file:
            problem_files = json.load(json_file)
        skip_files = [f['name'] for k, v in problem_files.items() for f in v]

        # Walk over .ann files to get names
        dcc_names = list()
        for ann_file in dcc_dir.rglob("*.ann"):
            name = Path(ann_file).stem
            if name not in skip_files:
                dcc_names.append(name)
        return dcc_names

    def shuffle_split_files(self, 
                            dcc_names: list,
                            n_splits: int = DEFAULT_N_SPLITS,
                            random_state: int = DEFAULT_RANDOM_STATE):

        # Shuffle & split files in train and test
        dcc_names = np.array(dcc_names)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_id = 0
        split_list = []
        for train, test in kf.split(dcc_names):
            split_list.append({'split_id': split_id,
                            'train': dcc_names[train].tolist(),
                            'test': dcc_names[test].tolist()})
            split_id += 1
        return split_list

    def write_splits(self,
                    split_list: list,
                    output_dir: Path = DEFAULT_OUTPUT_DIR):

        # Create output dir
        output_dir = output_dir
        output_dir.mkdir(exist_ok=True)

        # Write output file
        output_file = output_dir / 'split_list.json'
        with open(output_file, "w") as fp:
            json.dump(split_list, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--dcc_dir', type=Path, default=DEFAULT_DCC_DIR)
    parser.add_argument('--output_dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--skip_file', type=Path, default=DEFAULT_SKIP_FILE)
    parser.add_argument('--n_splits', type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument('--random_state', type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()
    
    splitter = DCCSplitter(args.dcc_dir, args.output_dir, args.skip_file, args.n_splits, args.random_state, write_to_file=True)
    splitter.split()