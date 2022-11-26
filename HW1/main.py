import pickle
import numpy as np
from os import path
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def calc_acc(pred_path, test_path, S):
    n = 0
    correct = 0
    pred_file = open(pred_path)
    test_file = open(test_path)
    pred_lines = pred_file.readlines()
    test_lines = test_file.readlines()
    total_true_tags = []
    total_predicted_tags = []
    for p_line, t_line in zip(pred_lines, test_lines):
        p_line = p_line[:-1] if p_line[-1] == "\n" else p_line
        t_line = t_line[:-1] if t_line[-1] == "\n" else t_line
        p_tags = [word_tag.split('_')[1] for word_tag in p_line.split(' ')]
        t_tags = [word_tag.split('_')[1] for word_tag in t_line.split(' ')]
        if len(p_tags) != len(t_tags):
            print('hi')
        n += len(p_tags)
        total_predicted_tags.extend(p_tags)
        total_true_tags.extend(t_tags)
        for p_tag, t_tag in zip(p_tags, t_tags):
            correct += int(p_tag == t_tag)
    S = list(S)
    conf_mat_norm = confusion_matrix(y_true=total_true_tags, y_pred=total_predicted_tags, labels=S, normalize='true')  # confusion matrix normalized to true tagging
    for i in range(conf_mat_norm.shape[0]):
        if conf_mat_norm[:, i].sum() == 0:
            conf_mat_norm[i, i] = 1
    top_confused = np.argpartition(np.array([1.0] * len(S)) - np.diag(conf_mat_norm), -10)[-10:]
    print("the following tags the model confused the most")
    for idx in top_confused:
        print(S[idx] + ' confused {}% of the time.'.format((1-conf_mat_norm[idx, idx]) * 100))
    conf_mat_norm_t = conf_mat_norm[top_confused, :]
    conf_mat_norm_t = conf_mat_norm_t[:, top_confused]
    plt.figure(dpi=1200)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat_norm_t, display_labels=np.array(S)[top_confused])
    disp.plot()

    plt.show()
    print(f'Accuracy is: {correct/n}')
    return correct/n

def create_dataset_for_small_models_eval(data_path, k):
    for i in range(k):
        data_file = open(data_path)
        data_lines = np.array(data_file.readlines())
        n = len(data_lines)
        test_indices = np.random.choice(n, int(n/5), replace=False)
        train_indices = np.delete(np.arange(n), test_indices)
        train = data_lines[train_indices]
        test = data_lines[test_indices]
        new_test = open(data_path.replace('/', '/'+str(i)+"_test_"), 'w+')
        new_train = open(data_path.replace('/', '/'+str(i)+"_train_"), 'w+')
        new_test.writelines(test)
        new_train.writelines(train)
def evaluate_small_model():
    threshold = 10
    lam = 10
    ret_val = 0
    k = 5
    train_path = "data/train2.wtag"
    if not path.exists("data/0_test_train2.wtag"):
        create_dataset_for_small_models_eval(train_path, k)
    for i in range():
        current_train_path = "data/train2.wtag".replace('/', '/'+str(i)+"_train_")
        current_test_path =  "data/train2.wtag".replace('/', '/'+str(i)+"_test_")

        weights_path = 'small_'+str(i)+'_weights.pkl'
        predictions_path = str(i)+'_predictions_train_2.wtag'

        statistics, feature2id = preprocess_train(current_train_path, threshold, is_model_2=True)
        get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

        with open(weights_path, 'rb') as f:
            optimal_params, feature2id = pickle.load(f)
        pre_trained_weights = optimal_params[0]
        tag_all_test(current_test_path, pre_trained_weights, feature2id, statistics, predictions_path)
        ret_val += calc_acc(predictions_path, current_test_path, statistics.tags)
    print(f'Total Avg. Accuracy is: {ret_val/k}')


def main():
    threshold = 1
    lam = 1
    is_model_2 = True
    #
    if is_model_2:
        evaluate_small_model()
    else:
        train_path = "data/train2.wtag"


        test_path = "data/comp1.words"

        weights_path = 'weights.pkl'
        predictions_path = 'comp_m1_203860721_308428127.wtag'
        # predictions_path = 'predictions_test_2.wtag'

        statistics, feature2id = preprocess_train(train_path, threshold, is_model_2=is_model_2)
        # get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

        with open(weights_path, 'rb') as f:
             optimal_params, feature2id = pickle.load(f)
        pre_trained_weights = optimal_params[0]
        #
        # print(pre_trained_weights)
        tag_all_test(test_path, pre_trained_weights, feature2id, statistics, predictions_path)
        #calc_acc(predictions_path, test_path, statistics.tags)

if __name__ == '__main__':
    main()
