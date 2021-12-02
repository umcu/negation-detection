#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 10:18:50 2021

@author: BasArends, Bram van Es
"""

import re
import torch
import codecs
import argparse
import numpy as np

from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, RobertaTokenizer, RobertaModel
from transformers import AdamW, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

import apex

class TextDatasetFromFiles(Dataset):
    """
       Loads a list of sentences into memory from a text file,
       split by newlines.
    """
    def __init__(self, input_files, tokenizer, args, skip_header = True):
        self.data = []
        self.y  = []
        #self.task = task
        for input_file in input_files:
            with codecs.open(input_file, 'r', encoding='utf8') as f:
                sentence = []
                labels = []
                if skip_header: next(f)

                for example in f.read().split("\n\n"):
                    lines = example.split("\n")
                    for i in range(len(lines)):
                    
                        line = lines[i].strip()
    
                        if line:
                            line = re.split('\t', line)
                            
                            if args.bio:
                                if line[1] != 'O': 
                                    if args.task == 'negation': tag = line[1] + '-' + line[2]
                                    elif args.task == 'experiencer': tag = line[1] + '-' + line[3]
                                    elif args.task == 'temporality': tag = line[1] + '-' + line[4]
                                else:
                                    tag = 'O'
                            else:
                                if args.task == 'negation': tag = line[2]
                                elif args.task == 'experiencer': tag = line[3]
                                elif args.task == 'temporality': tag = line[4]
                            
                            sentence.append(line[0])
                            labels.append(tag)
    
                        if not line or i == len(lines) - 1:
                            if len(sentence) > 0:
                                if len(tokenizer.tokenize(' '.join(sentence))) <= args.block_size - 2:
                                    self.data.append(' '.join(sentence))
                                    self.y.append(' '.join(labels))
                                else:
                                    length = 0
                                    sub_sentence, sub_labels = [], []
                                    for i in range(len(sentence)):
                                        if i != 0 and args.model_type == 'roberta':
                                            # add arbitrary token that will be not be split into multiple tokens and ignore it
                                            tokens = tokenizer.tokenize(". " + sentence[i])[1:]
                                        else:
                                            tokens = tokenizer.tokenize(sentence[i])
                                        
                                        if length + len(tokens) <= (args.block_size - 2) and i != len(sentence):
                                            sub_sentence.append(sentence[i])
                                            sub_labels.append(labels[i])
                                            length += len(tokens)
                                        else:
                                            self.data.append(' '.join(sub_sentence))
                                            self.y.append(' '.join(sub_labels))
                                            length = 0
                                            sub_sentence, sub_labels = [], []
                                    
    
                            sentence = []
                            labels = []


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.y[idx]

class TextDatasetFromDataFrame(Dataset):
    """
        Loads a list of sentences and labels for token-classification from a DataFrame
    """
    def __init__(self, df, tokenizer, args):
        self.data = []
        self.y  = []
        self.ids = []
        
        for id in df.Id.unique():
            lines = df[df.Id == id].values
            sentence = [row[1] for row in lines]
            if args.task=='negation':
                labels = [row[2]+'-'+row[3] if row[2] != 'O' else 'O' for row in lines]
            elif args.task=='experiencer':
                labels = [row[2]+'-'+row[4] if row[2] != 'O' else 'O' for row in lines]
            elif args.task=='temporality':
                labels = [row[2]+'-'+row[5] if row[2] != 'O' else 'O' for row in lines]
            _ids = [row[0]+'_'+row[6]+'_'+row[7] if row[2] != 'O' else 'O' for row in lines]

            if len(sentence) > 0:
                if len(tokenizer.tokenize(' '.join(sentence))) <= args.block_size - 2:
                    self.data.append(' '.join(sentence))
                    self.y.append(' '.join(labels))
                    self.ids.append(' '.join(_ids))
                else:
                    length = 0
                    sub_sentence, sub_labels = [], []
                    for i in range(len(sentence)):
                        if i != 0 and args.model_type == 'roberta':
                            # add arbitrary token that will be not be split into multiple tokens and ignore it
                            tokens = tokenizer.tokenize(". " + sentence[i])[1:]
                        else:
                            tokens = tokenizer.tokenize(sentence[i])
                        
                        if length + len(tokens) <= (args.block_size - 2) and i != len(sentence):
                            sub_sentence.append(sentence[i])
                            sub_labels.append(labels[i])
                            length += len(tokens)
                        else:
                            self.data.append(' '.join(sub_sentence))
                            self.y.append(' '.join(sub_labels))
                            self.ids.append(' '.join(_ids))
                            length = 0
                            sub_sentence, sub_labels = [], []
                            break
                
                
                
                

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.y[idx], self.ids[idx]

def create_batch(sentences, labels, tag2id, device, 
                    tokenizer, model_type, max_len, word_dropout=0.):
    """
    Converts a list of sentences to a padded batch of word ids. Returns
    an input batch, output tags, a sequence mask over the input batch,
    and a tensor containing the sequence length of each batch element.

    :param sentences: a list of sentences, each a list of token ids
    :param labels: a list of outputs
    :param tag2id: ids for the tags
    :param device:
    :param tokenizer:
    :param model_type:
    :param max_len:
    :param word_dropout: rate at which we omit words from the context (input)

    :returns: a batch of padded inputs, a batch of outputs, mask, lengths
    """

    # Tokenize input sentences
    tok = [[tokenizer.bos_token] + tokenizer.tokenize(sen) + [tokenizer.eos_token]  for sen in sentences]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tok]

    # Split tags    
    tags_ = [l.split() for l in labels]
    
    # Get pad id's
    pad_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]    
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Get label id's based on tags
    label_ids = []
    for labs, s  in zip(tags_, sentences):
        sent_labels = []
        
        words = s.split()
        for i in range(len(labs)):
            # roberta tokenizes the first token differently than others
            if i != 0 and model_type == 'roberta':
                # add arbitrary token that will be not be split into multiple tokens and ignore it
                tokens = tokenizer.tokenize(". " + words[i])[1:]
            else:
                tokens = tokenizer.tokenize(words[i])

            
            try:
                label_id = tag2id[labs[i]]
            except KeyError:
                label_id = pad_token_label_id

            sent_labels.extend([label_id] + [pad_token_label_id] * (len(tokens) - 1))
        
        label_ids.append([pad_token_label_id] + sent_labels + [pad_token_label_id])

    # Pad input and labels, create attention masks
    attention_masks = []
    for i in range(len(input_ids)):
        assert len(input_ids[i]) == len(label_ids[i]), f"Length of inputs and labels differs. Inputs:{input_ids[i]}, Labels: {label_ids[i]}"
        assert len(input_ids[i]) <= max_len, len(input_ids[i])
        attention_mask = [1] * len(input_ids[i])
        while len(input_ids[i]) < max_len:
            input_ids[i].append(pad_id)
            label_ids[i].append(pad_token_label_id)
            attention_mask.append(0)
        attention_masks.append(attention_mask)
    
    # Convert everything to PyTorch tensors.
    batch_input = torch.tensor(input_ids)
    batch_output = torch.tensor(label_ids)
    seq_mask = torch.tensor(attention_masks)

    # Move all tensors to the given device.
    batch_input = batch_input.to(device)
    batch_output = batch_output.to(device)
    seq_mask = seq_mask.to(device)

    return batch_input, batch_output, seq_mask

def eval_model(model, eval_dataset, tag2id, device, tokenizer, args, return_pred = False):
    """
    Computes loss and f1 score given dataset
    """
    
    dl = DataLoader(eval_dataset, batch_size = args.batch_size, shuffle=False)
    model.eval()
    pad_token_label_id = CrossEntropyLoss().ignore_index
    preds, out_label_ids = None, None
    id2tag = {v: k for k, v in tag2id.items()}
    
    with torch.no_grad():
        for sents, labs, ids in dl:
            
            x_in, y, seq_mask = create_batch(sents, labs, tag2id, device, tokenizer, 
                                             args.model_type, args.block_size)
            scores = model(x_in, attention_mask = seq_mask, labels = y)[1]
            scores = scores.detach().cpu().numpy()
            #label_ids = y.to('cpu').numpy()
            
            if preds is None:
                preds = scores
                out_label_ids = y.detach().cpu().numpy()
            else:
                preds = np.append(preds, scores, axis = 0)
                out_label_ids = np.append(out_label_ids, y.detach().cpu().numpy(), axis = 0)
                
    preds = np.argmax(preds, axis = 2)
    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    id_list = [[] for _ range(ids)]
    
    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i,j] != pad_token_label_id:
                out_label_list[i].append(id2tag[out_label_ids[i][j]])
                preds_list[i].append(id2tag[preds[i][j]])
                ids[i].append(ids[i][j])
    
    f = f1_score(out_label_list, preds_list)
    precision = precision_score(out_label_list, preds_list)
    recall = recall_score(out_label_list, preds_list)
    print('F1: %.3f ' % f)
    if args.do_print_class_report:
        print(classification_report(out_label_list, preds_list))
    
    
    if args.bootstrap:
        rng = np.random.RandomState(seed = 12345)
        idx = np.arange(len(out_label_list))

        precisions, recalls, f1s = [], [], []
        
        for i in range(args.bootsize):
            pred_idx = rng.choice(idx, size = len(idx), replace = True)
            boot_true, boot_pred = [], []
            for j in range(len(idx)):
                boot_true.append(out_label_list[pred_idx[j]])
                boot_pred.append(preds_list[pred_idx[j]])
            output = classification_report(boot_true, boot_pred, output_dict = True)
            if args.task == 'negation':
                precisions.append(output['Negated']['precision'])
                recalls.append(output['Negated']['recall'])
                f1s.append(output['Negated']['f1-score'])
            #TODO: other tasks
        
        print("Bootstrap confidence intervals for %s" % (args.task))
        print("Precision: %.2f (%.2f - %.2f)" % (np.mean(precisions), np.percentile(precisions, 2.5), np.percentile(precisions, 97.5)))
        print("Recall: %.2f (%.2f - %.2f)" % (np.mean(recalls), np.percentile(recalls, 2.5), np.percentile(recalls, 97.5)))
        print("F1: %.2f (%.2f - %.2f)" % (np.mean(f1s), np.percentile(f1s, 2.5), np.percentile(f1s, 97.5)))
                
    
    
    if return_pred:
        return f, precision, recall, preds_list, out_label_list, ids
    else:
        return f, precision, recall
                
def train_model(model, tokenizer, train_dataset, eval_dataset, tag2id, 
                        device, args, max_grad_norm = 1.0, amp=False):  
     dl = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
     best_model = args.output_dir + "best_model.pt" if args.output_dir.endswith("/") else args.output_dir + "/best_model.pt"
     print(model.config)
     
     
     no_decay = ["bias", "LayerNorm.weight"]
     optimizer_grouped_parameters = [
         {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.0},
         {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
     ]
     
     t_total = len(dl) // args.gradient_accumulation_steps * args.num_epochs
     optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr, eps = 1e-8)
     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps = t_total)
     
     if amp:
            model, optimizer = apex.amp.initialize(model, optimizer, opt_level = "O3")

     step = 0
     best_val_f = 0
     model.zero_grad()
     loss_eval_history = []

     for epoch_num in range(1, args.num_epochs + 1):
         tr_loss = 0
         epoch_examples, epoch_steps = 0, 0
         
         for sents, labs, _ in tqdm(dl, desc="Epoch %i" % epoch_num, position=0, leave=True):
             # set model in training mode and create batch
             model.train()
             x_in, y, seq_mask = create_batch(sents, labs, tag2id, device, tokenizer, args.model_type, args.block_size)
             
             # forward and backward pass
             loss = model(x_in, attention_mask = seq_mask, labels = y)[0]
             loss.backward()
             
             # track loss, mumber of examples and steps
             tr_loss += loss.item()
             epoch_examples += x_in.size(0)
             epoch_steps += 1
             

             if (step + 1) % args.gradient_accumulation_steps == 0:
                 # gradient clipping
                 torch.nn.utils.clip_grad_norm_(parameters = model.parameters(), max_norm = max_grad_norm)
             
                 # update parameters
                 optimizer.step()
                 scheduler.step()
                 model.zero_grad()
             
             # evaluate
             if args.do_eval and step % args.eval_steps == 0:
                 print("Epoch %d, step %d: training loss = %.2f" % (epoch_num, step, tr_loss/epoch_steps))
                
                 val_f, val_p, val_r = eval_model(model = model, eval_dataset = eval_dataset, tag2id = tag2id,
                                                 device = device, tokenizer = tokenizer, args = args)
                 loss_eval_history.append({'epoch': epoch_num, 'step': step, 
                                           'f1': val_f, 'precision': val_p, 
                                           'recall': val_r})
                 if val_f > best_val_f:
                     best_val_f = val_f
                     if args.do_write:
                        torch.save(model.state_dict(), best_model)
                        print("best_model f = %.3f" % best_val_f)
           
             step += 1
     
     print("Training finished, best model f = %.3f" % best_val_f)
     if args.save_model:
         model.save_pretrained(args.output_dir)
         tokenizer.save_pretrained(args.output_dir)

     if args.do_eval:
        return model, loss_eval_history
     else:
        return model, None
                
     

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type = str, required = True)
    parser.add_argument("--model_path", type = str, required = True)
    parser.add_argument("--tokenizer_path", default = None, type = str)
    parser.add_argument("--task", type = str, required = True)
    parser.add_argument("--train_data", type = str, default = None)
    parser.add_argument("--eval_data", type = str, default = None)
    parser.add_argument("--output_dir", type = str, default = None)
    parser.add_argument("--num_epochs", type = int, default = 4)
    parser.add_argument("--eval_steps", type = int, default = 100)
    parser.add_argument("--lr", type = float, default = 5e-5)
    parser.add_argument("--batch_size", type = int, default = 16)
    parser.add_argument("--gradient_accumulation_steps", type = int, default = 1)
    parser.add_argument("--block_size", type = int, default = 512)
    parser.add_argument("--bio", action = "store_true")
    parser.add_argument("--save_model", action = "store_true")
    parser.add_argument("--use_best_model", action = "store_true")
    parser.add_argument("--do_eval", action = "store_true")
    parser.add_argument("--do_train", action = "store_true")
    parser.add_argument("--evaluate_during_training", action = "store_true")
    parser.add_argument("--bootstrap", action = "store_true")
    parser.add_argument("--bootsize", type = int, default = 200)
    args = parser.parse_args()
    
    if args.task not in ['negation','temporality','experiencer']:
        raise ValueError("Unknown task \'%s\'" % args.task)
        
    if args.model_type not in ['roberta','bert',]:
        raise ValueError("Model type \'%s\' not supported" % args.model_type)
        
    if not args.do_eval and not args.do_train:
        raise ValueError("--do_train or --do_eval is required")
        
    if args.do_eval and not args.eval_data:
        raise ValueError("No eval_data supplied")
    
    if args.do_train and not args.train_data:
        raise ValueError("No train_data supplied")
        
    if (args.evaluate_during_training or args.save_model) and not args.output_dir:
        raise ValueError("Output directory missing")
    
    
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model_path
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    if args.bio:
        tag_ids = {'negation':{'B-Negated':0,'B-NotNegated':1,'I-Negated':2,'I-NotNegated':3},
                   'temporality':{'B-Recent':0,'B-Historical':1,'B-Hypothetical':2,'I-Recent':3,'I-Historical':4,'I-Hypothetical':5},
                   'experiencer':{'B-Patient':0,'B-Other':1,'I-Patient':2,'I-Other':3}}
    else:
        tag_ids = {'negation':{'Negated':0,'NotNegated':1},
                   'temporality':{'Recent':0,'Historical':1,'Hypothetical':2},
                   'experiencer':{'Patient':0,'Other':1}}
    tag2id = tag_ids[args.task]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path, num_labels = len(tag2id))
    if args.use_best_model:
        model.load_state_dict(torch.load(args.output_dir + 'best_model.pt'))
        
    model = model.to(device)

    if args.do_train:
        train_dataset = TextDatasetFromFiles([args.train_data], tokenizer, args)
    if args.do_eval or args.evaluate_during_training:
        eval_dataset = TextDatasetFromFiles([args.eval_data], tokenizer, args)
    else:
        eval_dataset = None
    
    if args.do_train:
        train_model(model = model, 
                    tokenizer = tokenizer, 
                    train_dataset = train_dataset, 
                    eval_dataset = eval_dataset, 
                    tag2id = tag2id, 
                    device = device, 
                    args = args)
    
    if args.do_eval:
        eval_model(model = model, 
                  tokenizer = tokenizer, 
                  eval_dataset = eval_dataset, 
                  tag2id = tag2id, 
                  device = device, 
                  args = args)
    
if __name__ == "__main__":
    main()
    
    
# cd /Users/BasArends/Documents/Studie/Medical\ Informatics/Internship/
# python3.7 ner/training.py --model_type roberta --model_path lm/robbert/robbert-v2-dutch-base --task negation --train_data ner/example_set.tsv --eval_data ner/example_set.tsv --num_epochs 20 --eval_steps 2 --batch_size 1 --bio
    
    
