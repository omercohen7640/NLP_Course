import numpy as np
import scipy.special
from scipy.sparse import csr_matrix

from preprocessing import read_test, represent_input_with_features
from tqdm import tqdm
import numpy

beam_width = 5

def memm_viterbi(sentence, pre_trained_weights, feature2id, statistics, beam_width=None):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    # TODO implement viterbi algorithm
    S = statistics.tags
    S = sorted(S)
    pi = numpy.zeros((len(sentence), len(S), len(S)))
    bp = numpy.zeros((len(sentence), len(S), len(S)))
    pi[0, S.index('*'), S.index('*')] = 1
    bp[0, S.index('*'), S.index('*')] = 1
    beam = S if beam_width is None else [('*','*')]
    distinct = S if beam_width is None else ['*']
    for idx in range(2, len(sentence)+1):
        for v in S:
            idxv = S.index(v)
            for u in distinct:
                idxu = S.index(u)
                probability = q_cal(S=S, sentence=sentence, pre_trained_weights=pre_trained_weights,
                                    feature2id=feature2id, v=v, u=u, idx=idx,beam=beam)
                mul = pi[idx - 2, :, idxu] * probability
                t = int(np.argmax(mul))
                bp[idx-1, idxu, idxv] = t
                pi[idx-1, idxu, idxv] = mul[t]
        beam = []
        ind = numpy.argpartition(pi[idx-1].flatten(), -beam_width)[-beam_width:]
        for i in ind:
            beam.append((S[int(i / len(S))], S[int(i % len(S))]))
        distinct = [element[1] for element in beam]
    tags = numpy.argmax(pi[pi.shape[0], :, :])
    tags = [int(tags / len(S)), int(tags % len(S))]
    for k in reversed(range(1, len(sentence) - 1)):
        tags.append(bp[k + 2, tags[k-2], tags[k-1]])
    return tags


def q_cal(S, sentence, pre_trained_weights, feature2id, v, u, idx, beam=None):
    n_features = feature2id.n_total_features
    binary_ind = []
    if beam is not None:
        S = []
        for i, j in beam:
            if j == u:
                S.append(i)
    for tag in S:
        zipped_list = zip(list(reversed(sentence[idx - 2:idx+1])), [v, u, tag])
        zipped_list_flat=[]
        for i, j in zipped_list:
            zipped_list_flat.extend([i,j])
        history = tuple(zipped_list_flat + [sentence[idx + 1]])
        binary_ind.append(represent_input_with_features(history, feature2id.feature_to_idx))
    f_xy = np.zeros((len(S), n_features))
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
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id, statistics, beam_width=beam_width)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()
