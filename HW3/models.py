import torch.nn
from torch import nn, Tensor
from torch.nn import functional as F
from chu_liu_edmonds import decode_mst


def eval_model(model, sentence):
    _, score_mat = model(sentence)
    predicted_tree = decode_mst(score_mat.detach().numpy(), score_mat.shape[-1], False)
    return predicted_tree


class DependencyParser(nn.Module):
    def __init__(self, embedding_dim, POS_dim, ratio=1, concate=True, num_layers=2):
        super(DependencyParser, self).__init__()
        self.embedding_dim = embedding_dim
        self.POS_dim = POS_dim
        self.hidden_dim = int(self.embedding_dim*ratio)
        self.POS_hidden_dim = int(self.POS_dim*ratio)
        self.concate = concate
        if self.concate:
            self.hidden_dim += self.POS_hidden_dim
            self.embedding_dim += self.POS_dim
            self.POS_hidden_dim = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_words = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=num_layers,
                                           batch_first=True, bidirectional=True, dtype=torch.float32)
        self.encoder_POS = None
        if self.concate:
            self.encoder_POS = torch.nn.LSTM(input_size=self.POS_dim, hidden_size=self.POS_dim, num_layers=num_layers,
                                             batch_first=True, bidirectional=True, dtype=torch.float32)
        self.edge_scorer = torch.nn.Linear((self.hidden_dim + self.POS_dim) * 4, 1, dtype=torch.float32)

    def forward(self, word_embed: torch.Tensor):
        if self.concate:
            model_out, _ = self.encoder_words(torch.concatenate(word_embed, dim=2).float())  # [batch_size, seq_len, hidden_dim*2]
        else:
            model_out_words, _ = self.encoder_words(word_embed[0].float())  # [batch_size, seq_len, hidden_dim*2]
            model_out_POS, _ = self.encoder_POS(word_embed[1].float())  # [batch_size, seq_len, hidden_dim*2]
            model_out = torch.concatenate([model_out_words, model_out_POS], 2)
        score_mat = torch.zeros([word_embed[0].shape[1]]*2)
        for idxi, i in enumerate(model_out[0]):
            for idxj, j in enumerate(model_out[0]):
                if idxi != idxj:
                    score_mat[idxi, idxj] = self.edge_scorer(torch.concatenate([i, j], dim=0))
        return score_mat




class GraphLoss(nn.NLLLoss):
    def __int__(self):
        super(GraphLoss, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        masked = F.log_softmax(input, dim=1) * target
        loss = - (torch.sum(masked) / torch.sum(target))
        return loss
