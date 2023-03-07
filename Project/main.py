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

train_path = './data/train.pkl'
val_path = './data/val.pkl'

dataset_dict = ['test',
                'comp']
encoder_types = ['small',
                'base',
                'large',
                '3b',
                '11b']


parser = argparse.ArgumentParser(description='Nimrod Admoni, nimrod216@gmail.com',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model_size', choices=encoder_types, required=False, default=encoder_types[0],
                    help='encoder architectures:\n' + ' | '.join(encoder_types))
parser.add_argument('--LR', default=0.1, type=float,
                    help='starting learning rate')
parser.add_argument('--LRD', default=0, type=int,
                    help='learning rate decay - if enabled LR is decreased')
parser.add_argument('--batch_size', default=32, type=int,
                    help='number of samples in mini batch')
parser.add_argument('--WD', default=0, type=float,
                    help='weight decay')
parser.add_argument('--MOMENTUM', default=0, type=float,
                    help='momentum')
parser.add_argument('--GAMMA', default=0.1, type=float,
                    help='gamma')
parser.add_argument('--epochs', default=None, type=int,
                    help='epochs number')
parser.add_argument('--seed', default=None, type=int,
                    help='seed number')
parser.add_argument('--device', default=None, type=str,
                    help='device type')
parser.add_argument('--model_path', default=None, help='model path to load')
parser.add_argument('--tag_only', default=0, type=int, help='do not run train, only tagging')
parser.add_argument('--v', default=0, type=int, help='verbosity level (0,1,2) (default:0)')
parser.add_argument('--port', default='12355', help='choose port for distributed run')

args = parser.parse_args()

t5_tokenizer = AutoTokenizer.from_pretrained("t5-"+args.model_size)


def preprocess_function(examples):

    inputs = [example[SRC_LANG] for example in examples["translation"]]
    targets = [example[TGT_LANG] for example in examples["translation"]]
    model_inputs = t5_tokenizer(inputs, max_length=128, truncation=True)

    with t5_tokenizer.as_target_tokenizer():
        labels = t5_tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

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



def train_network2(training_args, model, train_data, val_data, model_path=None):
    if model_path is None:
        data_collator = DataCollatorForSeq2Seq(tokenizer=t5_tokenizer,model=model)
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            tokenizer=t5_tokenizer,
            compute_metrics=compute_metrics2,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator,
        )
        #training
        bleu_acc = trainer.train()
        trainer.plot_results(header='trial_num{}'.format(0))
    val_tag = CustomDataset('./data/val.unlabeled')
    comp = CustomDataset('./data/comp.unlabeled')
    for data in [comp, val_tag]:
        tagging = model.generate(data)
        write_comp_file(tagging, data)

    
def model_init(trail):
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-"+args.model_size)
    model.config.max_length = 512
    return model

def optuna_hp_space(trial):
    hp_space = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        # "lr_scheduler_type ": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine","constant_with_warmup"]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'generation_num_beams': trial.suggest_int("generation_num_beams", 2, 5),
    }
    return hp_space


def main2():
    batch_size = args.batch_size
    wd = 1e-4
    epochs = 10
    beam = 3
    lr_scheduler_type = 'constant_with_warmup'
    training_args = Seq2SeqTrainingArguments(
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        output_dir="./",
        logging_steps=(batch_size*10),
        save_steps=10,
        eval_steps=4,
        weight_decay=wd,
        num_train_epochs=epochs,
        lr_scheduler_type='constant_with_warmup',
        save_strategy='epoch',
        save_total_limit=4,
        #group_by_length=True,
        predict_with_generate=True,
        generation_num_beams=beam,
        # use_mps_device=True,
        # logging_steps=1000,
        # save_steps=500,
        # eval_steps=7500,
        # warmup_steps=2000,
        # save_total_limit=3,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-"+args.model_size)
    # increase max length to generate longer sentences
    model.config.max_length = 512
    data = get_dataset_dict2()


    train_data = data['train'].map(preprocess_function, batched=True)
    val_data = data['val'].map(preprocess_function, batched=True)

    train_network2(training_args, model, train_data, val_data, args.model_path)




def parameter_search():
    batch_size = args.batch_size
    training_args = Seq2SeqTrainingArguments(
        lr_scheduler_type='constant_with_warmup',
        num_train_epochs=20,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        output_dir="./",
        logging_steps=(batch_size*10),
        save_steps=10,
        eval_steps=4,
        save_strategy='epoch',
        save_total_limit=4,
        predict_with_generate=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-"+args.model_size)
    # increase max length to generate longer sentences
    data = get_dataset_dict2()


    train_data = data['train'].map(preprocess_function, batched=True)
    val_data = data['val'].map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=t5_tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=None,
        model_init=model_init,
        args=training_args,
        tokenizer=t5_tokenizer,
        compute_metrics=compute_metrics2,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
    )
    best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=20)
    print(best_trial)






if __name__ == '__main__':
    parameter_search()
    # main2()
