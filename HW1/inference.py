from preprocessing import read_test
from tqdm import tqdm
import numpy

def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    #TODO implement viterbi algorithm
    pi = numpy.ones((len(sentence),S,S))
    for word, idx in enumerate(sentence):
        for v, idx1 in enumerate(S):
            for u, idx2 in enumerate(S):
                probability = pre_trained_weights*feature2id(sentence)
                pi[idx, idx1, idx2] = max(pi[idx-1]*probability)
    raise NotImplementedError


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()
