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
from preprocessing import get_dataset_dict, CustomDataset, get_dataset_dict2
import project_evaluate
from tqdm import tqdm

SRC_LANG = 'de'
TGT_LANG = 'en'
DEP = 'dep'


dataset_dict = ['test',
                'comp']
encoder_types = ['glove',
                 'word2vec',
                 'fastext']
t5_tokenizer = AutoTokenizer.from_pretrained("./val_model_checkpoint_43_7")

parser = argparse.ArgumentParser(description='Nimrod Admoni, nimrod216@gmail.com',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--batch_size', default=16, type=int,
                    help='number of samples in mini batch')
parser.add_argument('--model_path', default=None, help='model path to load')
parser.add_argument('--comp', default=0, type=int, help='do not run train, only tagging')
parser.add_argument('--v', default=0, type=int, help='verbosity level (0,1,2) (default:0)')
parser.add_argument('--port', default='12355', help='choose port for distributed run')


# def preprocess_function(examples):

#     inputs = [example[SRC_LANG] + example[DEP] for example in examples["translation"]]
#     # targets = [example[TGT_LANG] for example in examples["translation"]]
#     # model_inputs = t5_tokenizer(inputs, max_length=128, truncation=True)

#     with t5_tokenizer.as_target_tokenizer():
#         labels = t5_tokenizer(targets, max_length=128, truncation=True)

#     model_inputs["labels"] = labels["input_ids"]
#     return inputs


def write_comp_file(translated, lines_to_translate, file_name):
    f = open(file_name, 'w+')
    raise NotImplementedError
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


# def main():
#     args = parser.parse_args()
#     batch_size = args.batch_size
#     beam = 3
#     model = AutoModelForSeq2SeqLM.from_pretrained("args.model_path")
#     # increase max length to generate longer sentences
#     model.config.max_length = 512
#     data = get_dataset_dict2()
#     comp_data = data['comp']
#     val_data = data['val_untagged']
#     # data_collator = DataCollatorForSeq2Seq(tokenizer=t5_tokenizer, model=model)
#     if args.comp:
#         inference(model, comp_data, '')
#     else:
#         inference(model, val_data, '')
args = parser.parse_args()

def preprocess_function(examples):
    inputs = [example[SRC_LANG] + example[DEP] for example in examples["translation"]]
    #model_inputs = t5_tokenizer(inputs, max_length=128, truncation=True)
    return inputs

def get_word(dataset, index):
    return dataset.datasets_dict['test'].original_words[index]

def write_comp_file(tagging, dataset):
    #TODO: fix tagging function
    raise NotImplementedError
    name = './comp_203860721_308427128.labeled'
    f = open(name, 'w+')
    # untagged_words = dataset.datasets_dict['test'].deleted_word_index
    empty_lines = dataset.datasets_dict['test'].empty_lines
    assert (dataset.datasets_dict['test'].num_of_words >= len(tagging))

    tagging_index = 0
    for i in range(dataset.datasets_dict['test'].num_of_words):
        p, word, POS = get_word(dataset, i)
        if word == '':
            f.write(word + '\n')
        else:
            if i in empty_lines:
                f.write('{}\n'.format(word))
            else:
                tag = tagging[tagging_index]
                f.write('{}\t{}\t_\t{}\t_\t_\t{}\t_\t_\t_\n'.format(p, word, POS, int(tag)))
                tagging_index += 1
    f.write('\n')
    f.close()




def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

metric = datasets.load_metric("bleu")
def compute_metrics2(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, t5_tokenizer.pad_token_id)
    with t5_tokenizer.as_target_tokenizer():
        decoded_preds = t5_tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = t5_tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result =  project_evaluate.compute_metrics(decoded_preds,decoded_labels)
    return {'score': result}

bleu = evaluate.load("bleu")




def main():
    #model = AutoModelForSeq2SeqLM.from_pretrained("./val_model_checkpoint_43_7/pytorch_model.bin")
    generator = pipeline("text2text-generation", model="./val_model_checkpoint_43_7/", framework="pt", batch_size=8)
    # increase max length to generate longer sentences
    data = get_dataset_dict2()


    #comp_data = data['comp'].map(preprocess_function, batched=True)
    #val_data = data['val_unlabled'].map(preprocess_function, batched=True)
    # data_collator = DataCollatorForSeq2Seq(tokenizer=t5_tokenizer, model=model)
    # encoded_en = t5_tokenizer(comp_data[0],return_tensors="pt")
    translation = []
    val_true = []
    for val_u,val in tqdm(zip(data['val_unlabeled'],data['val'])):
        sentence = val_u['translation']['de']+val_u['translation']['dep']
        sentence_translation = generator(sentence)[0]['generated_text']
        translation.append(sentence_translation)
        val_true.append(val['translation']['en'])
    bleu_score = project_evaluate.compute_metrics(translation, val_true)
    
        
        
    
    



if __name__ == '__main__':
    main()
