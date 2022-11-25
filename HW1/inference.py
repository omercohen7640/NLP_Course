import numpy as np
import scipy.special
from scipy.sparse import csr_matrix
from scipy.sparse import csr_array
import time
from preprocessing import read_test, represent_input_with_features
from tqdm import tqdm
import numpy

beam_width = 2

def tag_all_test(test_path, pre_trained_weights, feature2id, statistics, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id, statistics, beam_width=beam_width)[2:]

        sentence = sentence[2:-1] # cut off the ~ and *
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()

def find_t_in_beam(S, u, beam):
    t_beam = []
    t_index = []
    for element in beam:
        if element[1] == u:
            t_beam.append(element[0])
            t_index.append(S.index(element[0]))
    return t_beam, t_index


def q_cal(S, sentence, pre_trained_weights, feature2id, v, u, current_word_idx, t_list):
    n_features = feature2id.n_total_features
    probabilities = []
    for t in t_list:
        binary_indexs = []
        for s in S:
            history = (sentence[current_word_idx],s, sentence[current_word_idx-1], u, sentence[current_word_idx-2], t,sentence[current_word_idx+1] )
            binary_indexs.append(represent_input_with_features(history, feature2id.feature_to_idx))
        f_xy = np.zeros((len(S), n_features))
        for i, indices in zip(range(len(S)), binary_indexs):
            f_xy[i, indices] = 1
        sparse_fxy = csr_matrix(f_xy)
        assert (len(pre_trained_weights) == n_features)
        probabilities.append(scipy.special.softmax(sparse_fxy.dot(pre_trained_weights))[S.index(v)])
    return numpy.array(probabilities)


def q_cal_efficient(S, sentence, pre_trained_weights, feature2id, v, u, current_word_idx, t_list):
    n_features = feature2id.n_total_features
    probabilities = []
    for t in t_list:
        col_indices = []
        row_indices = []
        for row_idx, s in enumerate(S):
            history = (sentence[current_word_idx],s, sentence[current_word_idx-1], u, sentence[current_word_idx-2], t,sentence[current_word_idx+1] )
            feature_vector = represent_input_with_features(history, feature2id.feature_to_idx)
            row_indices.extend(len(feature_vector)*[row_idx])
            col_indices.extend(feature_vector)
        data = len(row_indices)*[1]
        sparse_fxy = csr_array((data, (row_indices, col_indices)), shape=(len(S), n_features))
        # assert (len(pre_trained_weights) == n_features)
        probabilities.append(scipy.special.softmax(sparse_fxy.dot(pre_trained_weights))[S.index(v)])
    return numpy.array(probabilities)

def memm_viterbi(sentence, pre_trained_weights, feature2id, statistics, beam_width=None):
    n = len(sentence)  # includes the *,* and ~
    S = statistics.tags
    S = sorted(S)
    num_tags = len(S)
    pi = numpy.zeros((n + 1, num_tags, num_tags), dtype=np.double)
    bp = numpy.zeros((n + 1, num_tags, num_tags), dtype=np.int)
    pi[0, S.index('*'), S.index('*')] = 1
    bp[2, :, :] = S.index('*')
    beam = [('*', '*')]  # (t,u) for k=0
    for current_word_idx in range(2, n - 1):  # running from 2 -> (n-2)
        current_pi_idx = current_word_idx - 1
        for idxv, v in enumerate(S):
            u_beam = [element[1] for element in beam]
            for u in u_beam:
                idxu = S.index(u)
                t_list, t_index = find_t_in_beam(S, u, beam)
                # start = time.time()
                # q_vector = q_cal(S, sentence, pre_trained_weights, feature2id, v, u, current_word_idx, t_list)
                # end = time.time()
                # print(end-start)
                # start = time.time()
                q_vector_eff = q_cal_efficient(S, sentence, pre_trained_weights, feature2id, v, u, current_word_idx, t_list)
                # end = time.time()
                # print(end - start)
                # assert (q_vector_eff == q_vector).all()
                mul = pi[current_pi_idx - 1, t_index, idxu] * q_vector_eff
                bp[current_pi_idx, idxu, idxv] = t_index[np.argmax(mul)]
                pi[current_pi_idx, idxu, idxv] = np.max(mul)
        beam = []
        ind = numpy.argpartition(pi[current_pi_idx].flatten(), -beam_width)[-beam_width:]
        for i in ind:
            beam.append((S[int(i / len(S))], S[int(i % len(S))]))
    tags_index = (n - 1) * [0]  # we do not tag the last word (~) and the (*)
    argmax_pi_n = np.argmax(pi[current_pi_idx].flatten())
    tags_index[n - 3], tags_index[n - 2] = int(argmax_pi_n / len(S)), int(argmax_pi_n % len(S))
    for k in reversed(range(-1, current_pi_idx-1)):
        tags_index[k+1] = bp[k + 2, tags_index[k+2], tags_index[k+3]]
    tags = [S[t] for t in tags_index]
    return tags
