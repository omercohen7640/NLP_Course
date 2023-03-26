import datasets

from preprocessing import get_dataloader
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

SRC_LANG = 'de'
TGT_LANG = 'en'
DEP = 'dep'

parser = argparse.ArgumentParser(description='Nimrod Admoni, nimrod216@gmail.com',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--val',default=0, type=int, help='run val or not')

args = parser.parse_args()

def write_file(translated, lines_to_translate, file_name, start_lines):
    f = open(file_name, 'w+')
    orig_stack = 'German:\n'
    translated_stack = 'English:\n'
    for idx, line in enumerate(translated):
        if idx in start_lines:
            f.write(orig_stack + translated_stack + '\n')
            orig_stack = 'German:\n'
            translated_stack = 'English:\n'
        orig_stack += (line + '\n')
        translated_stack += (lines_to_translate[idx] + '\n')
    f.write('\n')
    f.close()

data = get_dataset_dict2()


def main_val():
    generator = pipeline("text2text-generation", model="./val_model_checkpoint/", framework="pt", batch_size=32,
                         device=0)
    translation = [out[0]["generated_text"] for out in tqdm(generator(KeyDataset(data['val_unlabeled']["translation"], "de")))]
    val_true = [val['translation']['en'] for val in data['val']]
    
    # for val_u,val in tqdm(zip(data['val_unlabeled'],)):
    #     sentence = val_u['translation']['de']+val_u['translation']['dep']
    #     sentence_translation = generator(sentence)[0]['generated_text']
    #     translation.append(sentence_translation)
    #     val_true.append()
    bleu_score = project_evaluate.compute_metrics(translation, val_true)
    print("BLEU score for val task is: {:.f2} .".format(bleu_score))
    write_file(translated=translation,lines_to_translate=data['val']['translation'],
               file_name='./val.labeled',start_lines=get_start_lines('./data/val.unlabeled'))
    
def main_comp():
    generator = pipeline("text2text-generation", model="./comp_model_checkpoint/", framework="pt", batch_size=4,
                         device=0)
    translation = []
    val_true = []
    for comp_line in tqdm(data['comp_unlabeled']):
        sentence = comp_line['translation']['de'] + comp_line['translation']['dep']
        sentence_translation = generator(sentence)[0]['generated_text']
        translation.append(sentence_translation)
    write_file(translated=translation,lines_to_translate=data['comp']['translation'],
               file_name='./comp.labeled',start_lines=get_start_lines('./data/comp.unlabeled'))

if __name__ == '__main__':
    get_text_from_file('./data/val.unlabeled',True)
    if args.val == 1:
        main_val()
    main_comp()
