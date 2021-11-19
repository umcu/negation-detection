# -*- coding: utf-8 -*-
"""
Created on Thu May  6 09:14:50 2021

@author: Bas.Arends
"""

import os
import sys
import argparse

def label_category(label):
    if label == 'Negated':
        return {'cat': 'neg', 'value': 1}
    elif label == 'NotNegated':
        return {'cat': 'neg', 'value': 2}
    elif label == 'Patient':
        return {'cat': 'exp', 'value': 1}
    elif label == 'Other':
        return {'cat': 'exp', 'value': 2}
    elif label == 'Recent':
        return {'cat': 'tem', 'value': 1}
    elif label == 'Historical':
        return {'cat': 'tem', 'value': 2}
    elif label == 'Hypothetical':
        return {'cat': 'tem', 'value': 3}
    else:
        raise ValueError("Unknown label: {}".replace("{}",label))

def get_dataset(textfiles, labelfiles, nrs):
    dataset = []
    for text, labels, nr in zip(textfiles, labelfiles, nrs):
        # Get labels for each character
        bio = [0] * len(text)
        neg = [0] * len(text)
        tem = [0] * len(text)
        exp = [0] * len(text)
        labels = labels.split('\n')
        
        for line in labels:
            if line == '': continue
            line = line.split('\t')
            try:
                label = line[1].split(' ')
            except IndexError:
                print(f"Error in {nr}, for line:{line}")
                break
            
            try:
                label_cat = label_category(label[0])
            except ValueError as e:
                print(f"Value Error: {e}, \n file: {nr}, \n line: {line}")
                break

            for i in range(int(label[1]), int(label[2])):
                bio[i] = 1
                if label_cat['cat'] == 'neg':
                    neg[i] = label_cat['value']
                elif label_cat['cat'] == 'exp':
                    exp[i] = label_cat['value']
                elif label_cat['cat'] == 'tem':
                    tem[i] = label_cat['value']
        
        # Transform from character-based to word-based
        data, first_word = [], True
        word, bio_tag, negation, experiencer, temporality = '', [], [], [], []
        
        for i in range(len(text)):
            c = text[i]
            if c in [' ','\n','.',',','!','?',':',';','-','/','\"','\'','(',')','#','>','<','+','=','|']:
                #if empty continue
                if max(len(set(bio_tag)), len(set(negation)), len(set(experiencer)), len(set(temporality))) == 0:
                    continue
                
                # if end of word: append to dataset
                elif max(len(set(bio_tag)), len(set(negation)), len(set(experiencer)), len(set(temporality))) == 1:
                    
                    # append data
                    data.append([nr, word, bio_tag[0], negation[0], experiencer[0], temporality[0]])
                    word, bio_tag, negation, experiencer, temporality = '', [], [], [], []
                    
                    # if medical term is finished, set to true, else false
                    if i != len(text) - 1:
                        first_word = bio[i] == 0
                        
                # mismatch in labels
                else:
                    print(f"Error in {nr}, for line:{line}")
                    #print(text, labels, nr)
                    #print(word)
                    #print(bio_tag, negation, experiencer, temporality)
                    break
                
                # add special characters separately
                if c in ['.',',','!','?',':',';','-','/','\'','\"','(',')','#','>','<','+','=','|']:
                    if not first_word and bio[i] == 1:
                        bio[i] = 2
                    data.append([nr, c, bio[i], neg[i], exp[i], tem[i]])
            
            # not end of word: collect labels of character
            else:
                if not first_word and bio[i] == 1:
                    bio[i] = 2
                word += c
                bio_tag.append(bio[i])
                negation.append(neg[i])
                experiencer.append(exp[i])
                temporality.append(tem[i])
        dataset.append(data)
    
    return dataset

def get_tsv(dataset):
    data_tsv = "Id\tWord\tBIO\tNegation\tExperiencer\tTemporality"
    for text in dataset:
        example = ""
        for line in text:
            if len(line) != 6:
                raise ValueError("Error in dataset")
            example += "\n" + line[0] + "\t"
            example += line[1] + "\t"
            for i in range(2,6):
                if line[i] == 0:
                    example += "O\t"
                elif line[i] == 1:
                    if i == 1: example += "B\t"
                    if i == 2: example += "Negated\t"
                    if i == 3: example += "Patient\t"
                    if i == 4: example += "Recent\t"
                elif line[i] == 2:
                    if i == 1: example += "I\t"
                    if i == 2: example += "NotNegated\t"
                    if i == 3: example += "Other\t"
                    if i == 4: example += "Historical\t"
                elif line[i] == 3 and i == 5:
                    example += "Hypothetical\t"
                else:
                    raise ValueError(f"Invalid label: {line[i]}")
        data_tsv += example.replace("\n\n", "\n") + "\n"
    return data_tsv

def main():
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--path', type=str, default='../EMCDutchClinicalCorpus', help='Path to data')
    path = parser.parse_args().path

    textfiles, labelfiles, nrs, filenames = [], [], [], []
    
    for directory in ['DL', 'GP', 'RD', 'SP']:
        for file in os.listdir(os.path.join(os.path.abspath(path), directory)):
            if file.endswith(".txt"):
                try:
                    filenames.append(file)
                    textfiles.append(open(os.path.join(path, directory, file), 'r', encoding = 'latin-1').read())
                    labelfiles.append(open(os.path.join(path, directory, file[:-4])+".ann", 'r', encoding = 'latin-1').read())
                    nrs.append(file[:-4])
                except UnicodeDecodeError as e:
                    print(f"Error:{e}; {file}")           
    
    dataset = get_dataset(textfiles, labelfiles, nrs)
    data_tsv = get_tsv(dataset)
    
    with open("DCC.tsv", 'w', encoding = 'utf-8') as f:
        for line in data_tsv:
            f.write(line)

    
if __name__ == '__main__':
    main()