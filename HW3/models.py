from torch import nn
from chu_liu_edmonds import decode_mst

class DependencyParser(nn.Module):
    def __init__(self, *args):
        super(DependencyParser, self).__init__()
        self.word_embedding = # Implement embedding layer for words (can be new or pretr
        self.hidden_dim = self.word_embedding.embedding_dim
        self.encoder = # Implement BiLSTM module which is fed with word embeddings and o
        self.edge_scorer = # Implement a sub-module to calculate the scores for all poss
        self.loss_function = # Implement the loss function described above
    def forward(self, sentence):
        word_idx_tensor, pos_idx_tensor, true_tree_heads = sentence
        # Pass word_idx through their embedding layer
        # Get Bi-LSTM hidden representation for each word in sentence
        # Get score for each possible edge in the parsing graph, construct score matrix

        # Calculate the negative log likelihood loss described above

        return loss, score_mat


    def eval_model(model, sentence):
        _, score_mat = model(sentence)
        predicted_tree = decode_mst(score_mat)
        return predicted_tree