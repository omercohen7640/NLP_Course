import numpy as np
import scipy.special
from scipy.sparse import csr_matrix

from preprocessing import read_test
from tqdm import tqdm
import numpy


def memm_viterbi(sentence, pre_trained_weights, feature2id, statistics):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    # TODO implement viterbi algorithm
    S = statistics.tags
    pi = numpy.zeros((len(sentence) + 1, len(S), len(S)))
    bp = numpy.zeros((len(sentence) + 1, len(S), len(S)))
    pi[0, '*', '*'] = 1
    bp[0, '*', '*'] = 1
    for idx in range(1, len(sentence) + 1):
        for idx1, v in enumerate(S):
            for idx2, u in enumerate(S):
                probability = q_cal(S=S, sentence=sentence, pre_trained_weights=pre_trained_weights,
                                    feature2id=feature2id, v=v, u=u, idx=idx)
                mul = pi[idx - 1, :, idx2] * probability
                bp[idx, idx2, idx1] = numpy.argmax(mul)
                pi[idx, idx2, idx1] = mul[bp[idx, idx2, idx1]]
    tags = numpy.argmax(pi[len(sentence)])
    tags = [int(tags / len(S)), int(tags % len(S))]
    for k in reversed(range(1, len(sentence) - 1)):
        tags.append(bp[k + 2, tags[-2], tags[-1]])
    return tags


def q_cal(S, sentence, pre_trained_weights, feature2id, v, u, idx):
    n_features = feature2id.n_total_features
    binary_ind = [feature2id.represent_input_with_features(
        zip(reversed(sentence[idx - 2:idx]), [v, u, max_tag]) + [sentence[idx + 1]],
        feature2id.feature_to_idx) for max_tag in S]
    f_xy = np.zeros(len(S), n_features)
    for i, indices in zip(range(len(S)), binary_ind):
        f_xy[i, indices] = 1
    sparse_fxy = csr_matrix(f_xy)
    assert (len(pre_trained_weights) == n_features)
    probability = scipy.special.softmax(sparse_fxy.dot(pre_trained_weights))
    return probability


def tag_all_test(test_path, pre_trained_weights, feature2id, statistics, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id, statistics)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()
