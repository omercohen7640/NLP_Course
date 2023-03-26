import datasets

from preprocessing import get_dataloader, GER_INIT, ENG_INIT
import Config as cfg
import pickle
import argparse
import sys
import os
from Trainer import NNTrainer
from models import *
import evaluate
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from preprocessing import get_dataset_dict, CustomDataset, get_dataset_dict2,get_start_lines, get_text_from_file
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import json
import project_evaluate

SRC_LANG = 'de'
TGT_LANG = 'en'
DEP = 'dep'

parser = argparse.ArgumentParser(description='Nimrod Admoni, nimrod216@gmail.com',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--val',default=0, type=int, help='run val or not')

args = parser.parse_args()

def write_file(translated, de_file, file_name):
    f = open(file_name, 'w+')
    orig_stack = 'German:\n'
    translated_stack = 'English:\n'
    german_sentences = []
    sen = ""
    with open(de_file,'r') as fd:
        to_copy = False
        for line in fd.readlines():
            line = line.strip()
            if line.startswith('Roots in English'):
                to_copy = False
                german_sentences.append(sen[:-1]) #we remove last space
                continue
            if to_copy:
                sen += line + " "
                continue
            if line == GER_INIT:
                to_copy = True
                sen = ""
    for en,de in zip(translated, german_sentences):
        f.write(orig_stack+de+'\n'+translated_stack+en+"\n\n")
    f.close()
     
    # for idx, line in enumerate(translated):
    #     f.write(orig_stack + data['val']['translation'][idx]['de'] + translated_stack + line +'\n' '\n')
    #     orig_stack = 'German:\n'
    #     translated_stack = 'English:\n'
    #     orig_stack += (line + '\n')
    #     translated_stack += (lines_to_translate[idx] + '\n')
    # f.write('\n')
    # f.close()

data = get_dataset_dict2()


def main_val():
    
    generator = pipeline("text2text-generation", model="./val_model/", framework="pt", batch_size=32,
                        device=0)
    translation = [out[0]["generated_text"] for out in tqdm(generator(KeyDataset(data['val_unlabeled'][0:10]["translation"], "de")))] #TODO: remove slicing on input data
    #val_true = [val['translation']['en'] for val in data['val']]
    
    # for val_u,val in tqdm(zip(data['val_unlabeled'],)):
    #     sentence = val_u['translation']['de']+val_u['translation']['dep']
    #     sentence_translation = generator(sentence)[0]['generated_text']
    #     translation.append(sentence_translation)
    #     val_true.append()
    # with open('./translation.dump','r') as f :
    #     translation = json.load(f)
    # bleu_score = project_evaluate.compute_metrics(translation, val_true)
    # print("BLEU score for val task is: {:.f2} .".format(bleu_score))
    write_file(translated=translation, de_file='./data/val.unlabeled', file_name='./val.labeled')

    blue = project_evaluate.calculate_score('./val.labeled','./data/val.labeled')
    return 0
    
def main_comp():
    generator = pipeline("text2text-generation", model="./comp_model_checkpoint/", framework="pt", batch_size=4,
                         device=0)
    translation = []
    val_true = []
    for comp_line in tqdm(data['comp_unlabeled']):
        sentence = comp_line['translation']['de'] + comp_line['translation']['dep']
        sentence_translation = generator(sentence)[0]['generated_text']
        translation.append(sentence_translation)
    write_file(translated=translation,
               file_name='./comp.labeled')

if __name__ == '__main__':
    get_text_from_file('./data/val.unlabeled',True)
    if args.val == 1:
        main_val()
    main_comp()
