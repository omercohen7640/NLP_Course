from typing import Optional, Any, Union
import numpy as np
import torch.nn
from torch import nn, Tensor
from torch.nn import functional as F
from transformers import EncoderDecoderModel, EncoderDecoderConfig, PretrainedConfig, BertConfig, GPT2Config, \
    AutoTokenizer, GPT2Tokenizer, DataCollatorForSeq2Seq, PreTrainedTokenizerBase
from collections import OrderedDict

from transformers.utils import PaddingStrategy


# def eval_model(model, sentence):
#     _, score_mat = model(sentence)
#     predicted_tree = decode_mst(score_mat.detach().numpy(), score_mat.shape[-1], False)
#     return predicted_tree


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



class EncDec(nn.Module):
    def __init__(self, enc="deepset/gbert-large", dec="bert-large-cased"):
        super(EncDec, self).__init__()
        self.enc_tokenizer = AutoTokenizer.from_pretrained(enc)
        self.dec_tokenizer = AutoTokenizer.from_pretrained(dec)
        self.create_encoder_config(enc)
        self.create_decoder_config(dec)
        self.enc_dec_config = EncoderDecoderConfig.from_encoder_decoder_configs(self.enc_config, self.dec_config)
        self.enc_dec_model = EncoderDecoderModel.from_encoder_decoder_pretrained(config=self.enc_dec_config,
                                                                                 encoder_pretrained_model_name_or_path=enc,
                                                                                 decoder_pretrained_model_name_or_path=dec)
        self.som = 1
        # self.enc_dec_model = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder=enc, decoder=dec)

    def forward(self, inc_inputs: torch.Tensor, dec_inputs: torch.Tensor):
        print('Forward Function is not implemented for CustomEncoderDecoder')
        raise NotImplementedError

    def create_encoder_config(self, name):
        enc_config = BertConfig()
        enc_config.name_or_path = name
        enc_config.is_encoder_decoder = True
        enc_config.is_decoder = False
        enc_config.eos_token_id = self.enc_tokenizer
        self.enc_config = enc_config

    def create_decoder_config(self, name):
        dec_config = BertConfig()
        dec_config.name_or_path = name
        dec_config.is_encoder_decoder = True
        dec_config.is_decoder = True
        dec_config.add_cross_attention = True
        dec_config.decoder_start_token_id = self.dec_tokenizer.bos_token
        self.dec_config = dec_config
