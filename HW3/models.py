import torch.nn
from torch import nn, Tensor
from torch.nn import functional as F
from chu_liu_edmonds import decode_mst
from collections import OrderedDict


def eval_model(model, sentence):
    _, score_mat = model(sentence)
    predicted_tree = decode_mst(score_mat.detach().numpy(), score_mat.shape[-1], False)
    return predicted_tree


class DependencyParser(nn.Module):
    def __init__(self, embedding_dim, POS_dim, vocab_size, POS_size, ratio=1, concate=True, num_layers=2, embed=False):
        super(DependencyParser, self).__init__()
        self.embedding_dim = embedding_dim
        self.POS_dim = POS_dim
        self.embed = embed
        self.vocab_size = vocab_size
        self.POS_size = POS_size

        # Embedder Init
        if self.embed:
            self.embedder = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
            self.POS_embedder = nn.Embedding(num_embeddings=self.POS_size, embedding_dim=self.POS_dim)
        self.hidden_dim = int(self.embedding_dim*ratio)
        self.POS_hidden_dim = int(self.POS_dim*ratio)
        self.concate = concate
        if self.concate:
            self.hidden_dim += self.POS_hidden_dim
            self.embedding_dim += self.POS_dim
            self.POS_dim = 0
            self.POS_hidden_dim = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # BiLSTM Init
        self.encoder_words = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=num_layers,
                                           batch_first=True, bidirectional=True)
        self.encoder_POS = None
        if not self.concate:
            self.encoder_POS = torch.nn.LSTM(input_size=self.POS_dim, hidden_size=self.POS_hidden_dim, num_layers=num_layers,
                                             batch_first=True, bidirectional=True)
        fc_in_size = (self.POS_hidden_dim + self.hidden_dim)*4
        self.edge_scorer = nn.Sequential(OrderedDict([('L1', nn.Linear(fc_in_size, int(fc_in_size/5))),
                                                      ('relu-1', nn.ReLU()),
                                                      ('L2', nn.Linear(int(fc_in_size/5), 100)),
                                                      ('relu-2', nn.ReLU()),
                                                      ('L3', nn.Linear(100, 1))]))

    def forward(self, inputs: torch.Tensor):

        n_words = inputs[0].shape[1]

        if self.concate:
            word_embed = inputs if not self.embed else torch.cat([self.embedder(inputs[0]), self.POS_embedder(inputs[1])], dim=2).float()
        else:
            word_embed = [None, None]
            word_embed[0] = inputs[0] if not self.embed else self.embedder(inputs[0])
            word_embed[1] = inputs[1] if not self.embed else self.POS_embedder(inputs[1])
        if self.concate:
            model_out, _ = self.encoder_words(word_embed)  # [batch_size, seq_len, hidden_dim*2]
        else:
            model_out_words, _ = self.encoder_words(word_embed[0].float())  # [batch_size, seq_len, hidden_dim*2]
            model_out_POS, _ = self.encoder_POS(word_embed[1].float())  # [batch_size, seq_len, hidden_dim*2]
            model_out = torch.cat([model_out_words, model_out_POS], 2)
        score_mat = torch.zeros([word_embed[0].shape[1]]*2, device=self.device)
        vec1 = model_out.squeeze().repeat([1, n_words]).reshape(-1, (self.POS_hidden_dim + self.hidden_dim)*2)
        vec2 = model_out.squeeze().repeat([n_words, 1])
        fc_input = torch.cat([vec1, vec2], dim=1)
        score_mat = self.edge_scorer(fc_input).reshape((n_words, n_words))
        return score_mat




class GraphLoss(nn.NLLLoss):
    def __int__(self):
        super(GraphLoss, self).__init__()

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        masked = F.log_softmax(inputs, dim=1) * target
        loss = - (torch.sum(masked) / torch.sum(target))
        return loss
