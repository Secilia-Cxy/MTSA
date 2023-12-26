import torch
import torch.nn as nn
from torch import optim

from src.layers.Embed import DataEmbedding, OutputEmbedding
from src.layers.SelfAttention_Family import FullAttention, AttentionLayer
from src.layers.Transformer_EncDec import Encoder, EncoderLayer
from src.models.DLbase import DLForecastModel


class Transformer(DLForecastModel):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.model = Model(args).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.d_layers = configs.d_layers

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.decoder = OutputEmbedding(configs.d_model, configs.c_out, self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.decoder(enc_out)

        dec_out = dec_out * stdev
        dec_out = dec_out + means
        return dec_out
