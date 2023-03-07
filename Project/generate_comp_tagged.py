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
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, AutoModelForSeq2SeqLM, AutoTokenizer
from preprocessing import get_dataset_dict, CustomDataset, get_dataset_dict2
import project_evaluate

SRC_LANG = 'de'
TGT_LANG = 'en'


dataset_dict = ['test',
                'comp']
encoder_types = ['glove',
                 'word2vec',
                 'fastext']
t5_tokenizer = AutoTokenizer.from_pretrained("t5-base")

parser = argparse.ArgumentParser(description='Nimrod Admoni, nimrod216@gmail.com',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--batch_size', default=16, type=int,
                    help='number of samples in mini batch')
parser.add_argument('--model_path', default=None, help='model path to load')
parser.add_argument('--comp', default=0, type=int, help='do not run train, only tagging')
parser.add_argument('--v', default=0, type=int, help='verbosity level (0,1,2) (default:0)')
parser.add_argument('--port', default='12355', help='choose port for distributed run')


def preprocess_function(examples):

    inputs = [example[SRC_LANG] for example in examples["translation"]]
    model_inputs = t5_tokenizer(inputs, max_length=128, truncation=True)

    with t5_tokenizer.as_target_tokenizer():
        labels = t5_tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def write_comp_file(translated, lines_to_translate, file_name):
    f = open(file_name, 'w+')
    start_lines = raise NotImplementedError
    tagging_index = 0
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


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def inference(model, lines_to_translate, file_name):
    translated = []
    for line in lines_to_translate:
        line_tokenized = t5_tokenizer(line, return_tensors="pt").input_ids
        output = model.generate(line_tokenized)
        out_line = t5_tokenizer.decode(output[0], skip_special_tokens=True)
        translated.append([out_line])
    write_comp_file(translated, lines_to_translate, file_name)


def main():
    args = parser.parse_args()
    batch_size = args.batch_size
    beam = 3
    model = AutoModelForSeq2SeqLM.from_pretrained("args.model_path")
    # increase max length to generate longer sentences
    model.config.max_length = 512
    data = get_dataset_dict2()
    comp_data = data['comp']
    val_data = data['val_untagged']
    # data_collator = DataCollatorForSeq2Seq(tokenizer=t5_tokenizer, model=model)
    if args.comp:
        inference(model, comp_data, '')
    else:
        inference(model, val_data, '')





if __name__ == '__main__':
    main()
