import torch
from torch import nn
import math
import torch_dct as dct
from einops import rearrange
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer, Decoder_change
from layers.SelfAttention_Family import ProbAttention, AttentionLayer


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.pred_len = configs.pred_len

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.Adropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.Edropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.patch_size = configs.patch_size
        self.stride = self.patch_size // 2
        num_patches = int((configs.seq_len - self.patch_size) / self.stride + 1)
        self.input_layer = nn.Linear(self.patch_size, configs.emb_dim)
        self.out_layer = nn.Linear(64 * num_patches, configs.pred_len)

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # TSLANET ------
        B, L, M = x_enc.shape  # B:64  L:500  M:1  ----
        x = rearrange(x_enc, 'b l m -> b m l')  # (64,500,1) --> (64,1,500)
        x = x.unfold(dimension=-1, size=self.patch_size,
                     step=self.stride)  # (64,1,14,64)  patch_size=64   stride=self.patch_size // 2
        x = rearrange(x, 'b m n p -> (b m) n p')  # (64,1,14,64) --> (64,14,64)
        z = self.input_layer(x)
        res = []


        #
        #
        #Due to the uncertainty of the review cycle, we have annotated the key parts of the code. If the paper is accepted, we will make it public.
        #
        #


        
        z, attns = self.encoder(z, attn_mask=None)
        outputs = self.out_layer(z.reshape(B * M, -1))  # (64,500)
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)  # (64,500) --> (64,500,1)

        return outputs

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)  # dec_out (64,500,1)

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]