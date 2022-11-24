import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test


def calc_acc(pred_path, test_path):
    n = 0
    correct = 0
    pred_file = open(pred_path)
    test_file = open(test_path)
    pred_lines = pred_file.readlines()
    test_lines = test_file.read_lines()
    for p_line,t_line in zip(pred_lines,test_lines):
        p_line = p_line[:-1] if p_line[-1] == "\n" else p_line
        t_line = t_line[:-1] if t_line[-1] == "\n" else t_line
        n += len(t_line)
        p_tags = [word_tag.split('_')[1] for word_tag in p_line.split(' ')]
        t_tags = [word_tag.split('_')[1] for word_tag in t_line.split(' ')]
        for p_tag, t_tag in zip(p_tags,t_tags):
            correct += int(p_tag == t_tag)
    print(f'Accuracy is: {correct/n}')
def main():
    threshold = 1
    lam = 1

    train_path = "data/train1.wtag"
    test_path = "data/train1.wtag"

    weights_path = 'weights.pkl'
    predictions_path = 'predictions.wtag'

    statistics, feature2id = preprocess_train(train_path, threshold)
    # get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    print(pre_trained_weights)
    tag_all_test(test_path, pre_trained_weights, feature2id, statistics, predictions_path)
    calc_acc(predictions_path,test_path)

if __name__ == '__main__':
    main()
