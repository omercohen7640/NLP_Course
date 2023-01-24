import torch.nn
from torch import nn, Tensor
from torch.nn import functional as F
from collections import OrderedDict


def eval_model(model, sentence):
    _, score_mat = model(sentence)
    predicted_tree = decode_mst(score_mat.detach().numpy(), score_mat.shape[-1], False)
    return predicted_tree


class CustomEncoderDecoder(nn.Module):
    def __init__(self, embedding_dim, vocab_size, num_layers=2, embed=False):
        super(CustomEncoderDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.embed = embed
        self.vocab_size = vocab_size

        # Embedder Init
        # TODO: fix embedder to be BERT
        self.embedder = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder Init
        self.encoder = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=num_layers,
                                     batch_first=True, bidirectional=True)
        # Decoder Init
        self.decoder = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=num_layers,
                                     batch_first=True, bidirectional=True)


    def forward(self, inputs: torch.Tensor):
        print('Forward Function is not implemented for CustomEncoderDecoder')
        raise NotImplementedError




class GraphLoss(nn.NLLLoss):
    def __int__(self):
        super(GraphLoss, self).__init__()

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        masked = F.log_softmax(inputs, dim=1) * target
        loss = - (torch.sum(masked) / torch.sum(target))
        return loss
