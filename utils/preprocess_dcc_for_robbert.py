# -*- coding: utf-8 -*-
"""
Created on Thu May  6 09:14:50 2021

@author: Bas.Arends, Bram van Es
"""

import os
import sys
import argparse
import re
import json
import pandas as pd
from pathlib import Path


def label_category(label):
    if label == "Negated":
        return {"cat": "neg", "value": 1}
    elif label == "NotNegated":
        return {"cat": "neg", "value": 2}
    elif label == "Patient":
        return {"cat": "exp", "value": 1}
    elif label == "Other":
        return {"cat": "exp", "value": 2}
    elif label == "Recent":
        return {"cat": "tem", "value": 1}
    elif label == "Historical":
        return {"cat": "tem", "value": 2}
    elif label == "Hypothetical":
        return {"cat": "tem", "value": 3}
    else:
        raise ValueError("Unknown label: {}".replace("{}", label))


def get_dataset(textfiles, labelfiles, nrs):
    dataset = []
    re_unicode = re.compile(r"[^\x00-\x7F]+")
    for text, labels, nr in zip(textfiles, labelfiles, nrs):
        text = re_unicode.sub("x", text)
        # Get labels for each character
        bio = [0] * len(text)
        neg = [0] * len(text)
        tem = [0] * len(text)
        exp = [0] * len(text)
        end = [0] * len(text)
        begin = [0] * len(text)

        labels = labels.split("\n")

        for line in labels:
            if line == "":
                continue
            line = line.split("\t")
            try:
                label = line[1].split(" ")
            except IndexError:
                print(f"Short line error in {nr}, for line:{line}")
                break

            try:
                label_cat = label_category(label[0])
            except ValueError as e:
                print(f"Value Error: {e}, \n file: {nr}, \n line: {line}")
                break

            try:
                for i in range(int(label[1]), int(label[2]) + 1):
                    bio[i] = 1
                    if label_cat["cat"] == "neg":
                        neg[i] = label_cat["value"]
                    elif label_cat["cat"] == "exp":
                        exp[i] = label_cat["value"]
                    elif label_cat["cat"] == "tem":
                        tem[i] = label_cat["value"]
                    end[i] = int(label[2])
                    begin[i] = int(label[1])
            except IndexError as e:
                print(f"Index Error, in {nr}, for line:{line}")
                break

        # Transform from character-based to word-based
        data, first_word = [], True
        word, bio_tag, negation, experiencer, temporality, ending, beginning = (
            "",
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for i in range(len(text)):
            c = text[i]
            if c in [
                " ",
                "\r",
                "\n",
                ".",
                ",",
                "!",
                "?",
                ":",
                ";",
                "-",
                "/",
                '"',
                "'",
                "(",
                ")",
                "#",
                ">",
                "<",
                "+",
                "=",
                "|",
                "[",
                "]",
            ]:
                # if empty continue
                if (
                    max(
                        len(set(bio_tag)),
                        len(set(negation)),
                        len(set(experiencer)),
                        len(set(temporality)),
                        len(set(beginning)),
                        len(set(ending)),
                    )
                    == 0
                ):
                    continue

                # if end of word: append to dataset
                elif (
                    max(
                        len(set(bio_tag)),
                        len(set(negation)),
                        len(set(experiencer)),
                        len(set(temporality)),
                        len(set(beginning)),
                        len(set(ending)),
                    )
                    == 1
                ):

                    # append data
                    data.append(
                        [
                            nr,
                            word,
                            bio_tag[0],
                            negation[0],
                            experiencer[0],
                            temporality[0],
                            beginning[0],
                            ending[0],
                        ]
                    )
                    (
                        word,
                        bio_tag,
                        negation,
                        experiencer,
                        temporality,
                        beginning,
                        ending,
                    ) = ("", [], [], [], [], [], [])

                    # if medical term is finished, set to true, else false
                    if i != len(text) - 1:
                        first_word = bio[i] == 0

                # mismatch in labels
                else:
                    print(f"Mismatch Error in {nr}, for line:{text[:i]}")
                    # print(text, labels, nr)
                    # print(word)
                    print(negation)
                    break

                # add special characters separately
                if c in [
                    ".",
                    ",",
                    "!",
                    "?",
                    ":",
                    ";",
                    "-",
                    "/",
                    "'",
                    '"',
                    "(",
                    ")",
                    "#",
                    ">",
                    "<",
                    "+",
                    "=",
                    "|",
                    "[",
                    "]",
                ]:
                    if (not first_word) & (bio[i] == 1):  # not first_word and
                        bio[i] = 2
                    data.append(
                        [nr, c, bio[i], neg[i], exp[i], tem[i], begin[i], end[i]]
                    )

            # not end of word: collect labels of character
            else:
                if (not first_word) & (bio[i] == 1):  # not first_word
                    bio[i] = 2
                word += c
                bio_tag.append(bio[i])
                negation.append(neg[i])
                experiencer.append(exp[i])
                temporality.append(tem[i])
                beginning.append(begin[i])
                ending.append(end[i])

        if len(data) == 0:
            print(f"EMPTY data for line: {nr}")
        else:
            # print(f"{nr} has {len(data)} data points")
            dataset.append(data)

    return dataset


def get_tsv(dataset):
    data_tsv = "Id\tWord\tBIO\tNegation\tExperiencer\tTemporality\tBegin\tEnd"
    re_multi_enter = re.compile(r"[\r\n]+")
    re_trail_tab = re.compile(r"[\t]+$")
    ids = []
    for text in dataset:
        example = ""
        for line in text:
            if len(line) != 8:
                raise ValueError("Error in dataset")
            ids.append(line[0])
            example += "\r\n" + line[0] + "\t"
            example += line[1] + "\t"
            for i in range(2, 8):
                if line[i] == 0:
                    example += "O"
                elif line[i] == 1:
                    if i == 2:
                        example += "B"
                    if i == 3:
                        example += "Negated"
                    if i == 4:
                        example += "Patient"
                    if i == 5:
                        example += "Recent"
                elif line[i] == 2:
                    if i == 2:
                        example += "I"
                    if i == 3:
                        example += "NotNegated"
                    if i == 4:
                        example += "Other"
                    if i == 5:
                        example += "Historical"
                elif line[i] == 3 and i == 5:
                    example += "Hypothetical"
                elif (i == 6) | (i == 7):
                    example += str(line[i])
                else:
                    raise ValueError(f"Invalid label: {line[i]}")
                if i < 7:
                    example += "\t"
        example = re_trail_tab.sub("", example)
        data_tsv += re_multi_enter.sub("\r\n", example)
    return data_tsv, ids


def get_dataframe(dataset):
    data_tsv = []
    re_multi_enter = re.compile(r"[\r\n]+")
    re_trail_tab = re.compile(r"[\t]+$")
    ids = []
    for text in dataset:
        example = ""
        for line in text:
            if len(line) != 8:
                raise ValueError("Error in dataset")
            ids.append(line[0])
            line_dict = {}
            line_dict["Id"] = line[0]
            line_dict["Word"] = line[1]
            for i in range(2, 8):
                if (line[i] == 0) and (i < 6):
                    line_dict["BIO"] = "O"
                    line_dict["Negation"] = "O"
                    line_dict["Experiencer"] = "O"
                    line_dict["Temporality"] = "O"
                    line_dict["Begin"] = "O"
                    line_dict["End"] = "O"
                elif (line[i] == 1) and (i < 6):
                    if i == 2:
                        line_dict["BIO"] = "B"
                    if i == 3:
                        line_dict["Negation"] = "Negated"
                    if i == 4:
                        line_dict["Experiencer"] = "Patient"
                    if i == 5:
                        line_dict["Temporality"] = "Recent"
                elif (line[i] == 2) and (i < 6):
                    if i == 2:
                        line_dict["BIO"] = "I"
                    if i == 3:
                        line_dict["Negation"] = "NotNegated"
                    if i == 4:
                        line_dict["Experiencer"] = "Other"
                    if i == 5:
                        line_dict["Temporality"] = "Historical"
                elif (line[i] == 3) and (i == 5):
                    line_dict["Temporality"] = "Hypothetical"
                elif i == 6:
                    line_dict["Begin"] = str(line[i])
                elif i == 7:
                    line_dict["End"] = str(line[i])
                else:
                    raise ValueError(f"Invalid label: {line[i]}")
            data_tsv.append(line_dict)

    return pd.DataFrame(data_tsv), ids


def main():
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument(
        "--path",
        type=str,
        default="../data/EMCDutchClinicalCorpus",
        help="Path to data",
    )
    directory = os.path.dirname(__file__)
    path = Path(os.path.join(directory, parser.parse_args().path))

    textfiles, labelfiles, nrs, filenames = [], [], [], []
    # ignore_list = ['GP2941', 'GP1832', 'GP2687', 'GP1971', 'GP1918', 'GP2007', 'GP1902', 'GP2282', 'GP1260', 'GP1448', 'GP3084', 'GP2351',
    #               'GP2188', 'GP2175', 'GP2392', 'GP2588', 'GP1823', 'GP1625', 'GP2435', 'GP2570', 'GP1757', 'GP2729', 'GP2476', 'GP2586',
    #               'GP1122', 'GP2252', 'GP2072', 'GP3035']
    exclusion_list_from_file = [
        d["name"]
        for d in json.load(open(path / "DCC_files_to_exclude.json"))[
            "annotation_errors"
        ]
    ]
    exclusion_list_from_file += [
        d["name"]
        for d in json.load(open(path / "DCC_files_to_exclude.json"))["corrupted"]
    ]

    exclusion_list_from_file += [
        d["name"] for d in json.load(open(path / "DCC_files_to_exclude.json"))["empty"]
    ]

    ignore_list = set(exclusion_list_from_file)
    print(f"Ignoring {len(ignore_list)} files")

    count = 0
    for directory in ["DL", "GP", "RD", "SP"]:
        for file in os.listdir(os.path.join(os.path.abspath(path), directory)):
            if (file.endswith(".txt")) & (file.split(".")[0] not in ignore_list):
                try:
                    filenames.append(file)
                    textfiles.append(
                        open(
                            os.path.join(path, directory, file), "r", encoding="latin-1"
                        ).read()
                    )
                    labelfiles.append(
                        open(
                            os.path.join(path, directory, file[:-4]) + ".ann",
                            "r",
                            encoding="latin-1",
                        ).read()
                    )
                    nrs.append(file[:-4])
                    count += 1
                except UnicodeDecodeError as e:
                    print(f"Error:{e}; {file}")

    print(f"\t{count} acceptable files found, {len(set(nrs))} unique files")
    dataset = get_dataset(textfiles, labelfiles, nrs)
    data_tsv, ids = get_tsv(dataset)
    print(f"\tProcessed {len(set(ids))} files")
    with open("DCC.tsv", "w", encoding="utf-8") as f:
        for line in data_tsv:
            f.write(line)

    df, ids = get_dataframe(dataset)
    print(f"\tProcessed {len(set(ids))} files")
    df.to_csv("DCC_df.csv", index=False, sep="\t")


if __name__ == "__main__":
    main()
