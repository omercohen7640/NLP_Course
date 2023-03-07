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
encoder_types = ['glove',
                 'word2vec',
                 'fastext']
t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")

parser = argparse.ArgumentParser(description='Nimrod Admoni, nimrod216@gmail.com',
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-ts', '--test_set', metavar='DATASET', choices=dataset_dict, required=False,
                    help='model dataset:\n' + ' | '.join(dataset_dict))
parser.add_argument('-e', '--encoder', metavar='enc', choices=encoder_types, required=False,
                    help='encoder architectures:\n' + ' | '.join(encoder_types))
parser.add_argument('--LR', default=0.1, type=float,
                    help='starting learning rate')
parser.add_argument('--LRD', default=0, type=int,
                    help='learning rate decay - if enabled LR is decreased')
parser.add_argument('--batch_size', default=16, type=int,
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




# def objective(trial):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     cfg.LOG.write("running on " + device)
#     ephocs = trial.suggest_int('ephocs', low=25, high=50)
#     # ephocs = 1
#     batch_size = trial.suggest_int('batch_size', low=3, high=7)
#     # batch_size = 5
#     lr = trial.suggest_loguniform('lr', 5e-5, 1e-2)
#     # lr = 0.01
#     # embedder = trial.suggest_categorical('embedder',['glove','word2vec','fasttext'])
#     embedder = 'custom'
#     wd = trial.suggest_loguniform('wd', 1e-5, 1e-3)
#     lmbda = trial.suggest_loguniform('lmbda', 1e-5, 0.1)
#     concat = True  # 1 = concat, 0 = no_concat
#     # concat = 0
#     lstm_layer_num = trial.suggest_int('lstm_layer_n', low=2, high=4)
#     # lstm_layer_num = 2
#     ratio = trial.suggest_float('ratio', low=0.5, high=1)
#     embedding_dim = trial.suggest_int('embedding_dim', low=80, high=150)
#     pos_dim = trial.suggest_int('pos_dim', low=15, high=40)
#     dataset = load_dataset(encoder=embedder)
#     # print(f'ephocs={ephocs}, batch size={2**batch_size}, lr={lr}, wd_size={wd}')
#     uas = train_network(dataset=dataset, epochs=ephocs, batch_size=2 ** batch_size, trial_num=trial.number,
#                         seed=None, LR=lr, LRD=0, WD=wd, MOMENTUM=0, GAMMA=0.1, lmbda=lmbda,
#                         device=device, save_all_states=True, model_path=None, test_set='test', concat=concat,
#                         lstm_layer_n=lstm_layer_num, ratio=ratio, embedding_dim=embedding_dim, POS_dim=pos_dim)
#     return uas
#
# def parameter_sweep():
#     cfg.LOG.start_new_log(name='parameter_search')
#     study = optuna.create_study(direction='maximize')
#     study.optimize(objective, n_trials=50)


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

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    return torch.from_numpy(bleu.compute(predictions=pred_ids, references=labels_ids))

def train_network(training_args, model, train_data, val_data, model_path=None):
    if model_path is None:
        data_collator = DataCollatorForSeq2Seq(tokenizer=model.enc_tokenizer,model=model.enc_dec_model)
        trainer = Seq2SeqTrainer(
            model=model.enc_dec_model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator,
            predict_with_generate=True,
        )
        #training
        bleu_acc = trainer.train()
        trainer.plot_results(header='trial_num{}'.format(0))
    val_tag = CustomDataset('./data/val.unlabeled')
    comp = CustomDataset('./data/comp.unlabeled')
    for data in [comp, val_tag]:
        tagging = model.generate(data)
        write_comp_file(tagging, data)

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

def main():
    args = parser.parse_args()
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
        lr_scheduler_type=lr_scheduler_type,
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

    model = EncDec()
    # if os.path.exists(train_path):
    #    with open(train_path, 'rb') as f:
    #        train_data = pickle.load(f)
    #    with open(val_path, 'rb') as f:
    #        val_data = pickle.load(f)
    # else:
        # train_data = CustomDataset(path='./data/train.labeled')
        # val_data = CustomDataset(path='./data/val.labeled')

        #with open(train_path, 'wb') as f:
        #    pickle.dump(train_data, f)
        #with open(val_path, 'wb') as f:
        #    pickle.dump(val_data, f)
    #train_network(training_args, model, data['train'], data['val'], args.model_path)
    # if os.path.exists(train_path):
    #     with open(train_path, 'rb') as f:
    #         train_data = pickle.load(f)
    #     with open(val_path, 'rb') as f:
    #         val_data = pickle.load(f)
    # else:
    #     train_data = CustomDataset(path='./data/train.labeled')
    #     val_data = CustomDataset(path='./data/val.labeled')
    #     with open(train_path, 'wb') as f:
    #         pickle.dump(train_data, f)
    #     with open(val_path, 'wb') as f:
    #         pickle.dump(val_data, f)
    enc_tokenizer = model.enc_tokenizer
    dec_tokenizer = model.dec_tokenizer
    encoder_max_length = 512
    decoder_max_length = 512
    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        inputs = enc_tokenizer([e['de'] for e in batch['translation']], padding="max_length", truncation=True, max_length=encoder_max_length)
        outputs = dec_tokenizer([e['en'] for e in batch['translation']], padding="max_length", truncation=True, max_length=decoder_max_length)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [[-100 if token == enc_tokenizer.pad_token_id else token for token in labels] for labels in
                           batch["labels"]]

        return batch

    data = get_dataset_dict(model.enc_tokenizer, model.dec_tokenizer)
    train_data = data['train'].map(process_data_to_model_inputs,
                                    batched=True,
                                    batch_size=batch_size*100,
                                   )

    val_data = data['val'].map(process_data_to_model_inputs,
                            batched=True,
                            batch_size=batch_size*100,
                            )


    train_network(training_args, model, train_data, val_data, args.model_path)



def main2():
    args = parser.parse_args()
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
        lr_scheduler_type=lr_scheduler_type,
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

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    # increase max length to generate longer sentences
    model.config.max_length = 512
    data = get_dataset_dict2()


    train_data = data['train'].map(preprocess_function, batched=True)
    val_data = data['val'].map(preprocess_function, batched=True)

    train_network2(training_args, model, train_data, val_data, args.model_path)





if __name__ == '__main__':
    main2()
