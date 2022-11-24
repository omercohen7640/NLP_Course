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
    # pi = numpy.zeros((len(sentence), len(S), len(S)), dtype=int)
    pi = numpy.zeros((len(sentence) - 1, len(S), len(S)), dtype=np.double)
    bp = numpy.zeros((len(sentence) - 1, len(S), len(S)), dtype=int)
    pi[0, S.index('*'), S.index('*')] = 1
    bp[0, S.index('*'), S.index('*')] = S.index('*')
    beam = S if beam_width is None else [('*', '*')]
    distinct = S if beam_width is None else ['*']
    for idx in range(2, len(sentence) - 1):
        for v in S:
            idxv = S.index(v)
            for u in distinct:
                idxu = S.index(u)
                probability = q_cal(S=S, sentence=sentence, pre_trained_weights=pre_trained_weights,
                                    feature2id=feature2id, v=v, u=u, idx=idx, beam=beam)
                indices_to_zero = []
                t_beam = [element[0] for element in beam]
                for s in S:
                    if s not in t_beam:
                        indices_to_zero.append(S.index(s))
                probability[indices_to_zero] = 0
                mul = pi[idx - 2, :, idxu] * probability
                t = int(np.argmax(mul))
                bp[idx - 1, idxu, idxv] = t
                pi[idx - 1, idxu, idxv] = mul[t]
        beam = []
        ind = numpy.argpartition(pi[idx - 1].flatten(), -beam_width)[-beam_width:]
        mask = np.zeros((len(S), len(S)))
        for i in ind:
            beam.append((S[int(i / len(S))], S[int(i % len(S))]))
            mask[int(i / len(S)), int(i % len(S))] = 1
        pi[idx - 1, :, :] = pi[idx - 1, :, :] * mask
        distinct = [element[1] for element in beam]
    # before_last_word_argmax = numpy.argmax(pi[pi.shape[0]-1, :, S.index('~')])
    before_last_word_argmax = numpy.argmax(pi[pi.shape[0] - 1, :, :])
    tags_id = len(sentence) * [0]
    tags_id[-2], tags_id[-1] = (before_last_word_argmax, S.index('~'))
    for k in reversed(range(1, len(sentence) - 2)):
        tags_id[k] = int(bp[k + 2, tags_id[k + 1], tags_id[k + 2]])
    tags = [S[i] for i in tags_id]
    return tags[:-1]

def q_cal(S, sentence, pre_trained_weights, feature2id, v, u, idx, beam=None):
    n_features = feature2id.n_total_features
    binary_ind = []
    # if beam is not None:
    #     S = []
    #     for i, j in beam:
    #         if j == u:
    #             S.append(i)
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

        sentence = sentence[2:-1]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()





def memm_viterbi_orig(sentence, pre_trained_weights, feature2id, statistics, beam_width=None):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    # TODO implement viterbi algorithm
    S = statistics.tags
    S = sorted(S)
    # pi = numpy.zeros((len(sentence), len(S), len(S)), dtype=int)
    pi = numpy.zeros((len(sentence)-1, len(S), len(S)), dtype=np.double)
    bp = numpy.zeros((len(sentence)-1, len(S), len(S)), dtype=int)
    pi[0, S.index('*'), S.index('*')] = 1
    bp[0, S.index('*'), S.index('*')] = 1
    beam = S if beam_width is None else [('*','*')]
    distinct = S if beam_width is None else ['*']
    for idx in range(2, len(sentence)-1):
        for v in S:
            idxv = S.index(v)
            for u in distinct:
                idxu = S.index(u)
                probability = q_cal(S=S, sentence=sentence, pre_trained_weights=pre_trained_weights,
                                    feature2id=feature2id, v=v, u=u, idx=idx, beam=beam)
                indices_to_zero = []
                t_beam = [element[0] for element in beam]
                for s in S:
                    if s not in t_beam:
                        indices_to_zero.append(S.index(s))
                probability[indices_to_zero] = 0
                mul = pi[idx - 2, :, idxu] * probability
                t = int(np.argmax(mul))
                bp[idx-1, idxu, idxv] = t
                pi[idx-1, idxu, idxv] = mul[t]
        beam = []
        ind = numpy.argpartition(pi[idx-1].flatten(), -beam_width)[-beam_width:]
        mask = np.zeros((len(S),len(S)))
        for i in ind:
            beam.append((S[int(i / len(S))], S[int(i % len(S))]))
            mask[int(i / len(S)), int(i % len(S))] = 1
        pi[idx-1, :, :] = pi[idx-1, :, :]*mask
        distinct = [element[1] for element in beam]
    # before_last_word_argmax = numpy.argmax(pi[pi.shape[0]-1, :, S.index('~')])
    before_last_word_argmax = numpy.argmax(pi[pi.shape[0]-1, :, :])
    tags_id = len(sentence)*[0]
    tags_id[-2], tags_id[-1] = (before_last_word_argmax, S.index('~'))
    for k in reversed(range(1, len(sentence)-2)):
        tags_id[k] = int(bp[k+2, tags_id[k+1], tags_id[k+2]])
    tags = [S[i] for i in tags_id]
    return tags[:-1]


def memm_viterbi_tuple(sentence, pre_trained_weights, feature2id, statistics, beam_width=None):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    # TODO implement viterbi algorithm
    S = statistics.tags
    S = sorted(S)
    # pi = numpy.zeros((len(sentence), len(S), len(S)), dtype=int)
    pi = numpy.zeros((len(sentence)-2, len(S), len(S)), dtype=np.double)
    bp = numpy.zeros((len(sentence)-2, len(S), len(S)), dtype=int)
    pi[0, S.index('*'), S.index('*')] = 1
    bp[0, S.index('*'), S.index('*')] = 1
    beam = S if beam_width is None else [('*','*')]
    distinct = S if beam_width is None else ['*']
    for tuple_num in range(1, len(sentence) - 2):
        for v in S:
            idxv = S.index(v)
            for u in distinct:
                idxu = S.index(u)
                current_word_idx = tuple_num + 1
                probability = q_cal(S=S, sentence=sentence, pre_trained_weights=pre_trained_weights,
                                    feature2id=feature2id, v=v, u=u, idx=current_word_idx, beam=beam)
                indices_to_zero = []
                t_beam = [element[0] for element in beam]
                for s in S:
                    if s not in t_beam:
                        indices_to_zero.append(S.index(s))
                probability[indices_to_zero] = 0
                mul = pi[tuple_num - 1, :, idxu] * probability
                t = int(np.argmax(mul))
                bp[tuple_num, idxu, idxv] = t
                pi[tuple_num, idxu, idxv] = mul[t]
        beam = []
        ind = numpy.argpartition(pi[tuple_num].flatten(), -beam_width)[-beam_width:]
        mask = np.zeros((len(S),len(S)))
        for i in ind:
            beam.append((S[int(i / len(S))], S[int(i % len(S))]))
            mask[int(i / len(S)), int(i % len(S))] = 1
        pi[tuple_num, :, :] = pi[tuple_num, :, :]*mask
        distinct = [element[1] for element in beam]
    # before_last_word_argmax = numpy.argmax(pi[pi.shape[0]-1, :, S.index('~')])
    # TODO make sure tag values are correct
    before_last_word_argmax = numpy.argmax(pi[pi.shape[0] - 1, :, :])
    tags_id = (len(sentence) - 1) * [0]
    tags_id[-2], tags_id[-1] = (int(before_last_word_argmax / len(S)), int(before_last_word_argmax % len(S)))
    for k in reversed(range(1, len(sentence) - 2)):
        tags_id[k - 1] = int(bp[k, tags_id[k], tags_id[k + 1]])
    tags = [S[i] for i in tags_id]
    return tags[:-1]


def memm_viterbi_new(sentence, pre_trained_weights, feature2id, statistics, beam_width=None):
    n = len(sentence) # includes the *,* and ~
    S = statistics.tags
    S = sorted(S)
    num_tags = len(S)
    pi = numpy.zeros((n+1, num_tags, num_tags), dtype=np.double)
    bp = numpy.zeros((n+1, num_tags, num_tags), dtype=np.int)
    pi[0, S.index('*'), S.index('*')] = 1
    bp[2, :, :] = S.index('*')
    beam = [('*', '*')]  # (t,u) for k=0
    for current_word_idx in range(2, n-1):  # running from 2 -> (n-2)
        current_pi_idx = current_word_idx - 1
        for idxv, v in enumerate(S):
            u_beam = [element[1] for element in beam]
            for u in u_beam:
                idxu = S.index(u)
                t_beam = find_t_in_beam(u,beam)
                for t in t_beam:
                    idxt = S.index(t)



def find_t_in_beam(u,beam):
    t_beam = []
    for element in beam:
        if element[1] == u:
            t_beam.append(element[0])
    return t_beam

def q_cal_new(S, sentence, pre_trained_weights, feature2id, v, u,t, current_word_idx, beam=None):
    n_features = feature2id.n_total_features

    for tag in S:

        history = (sentence[current_word_idx-2], t, sentence[current_word_idx-1], u, sentence[current_word_idx], v, sentence[current_word_idx+1])

