import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=1, groups=c_in)  # WFTC的使用组卷积

        self.norm = nn.LayerNorm(c_in)
        self.activation = nn.SELU()

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.norm(x)
        x = self.activation(x)

        return x
# class ConvLayer(nn.Module):
#     def __init__(self, c_in):
#         super(ConvLayer, self).__init__()
#         self.downConv = nn.Conv1d(in_channels=c_in,
#                                   out_channels=c_in,
#                                   kernel_size=3,
#                                   padding=2,
#                                   padding_mode='circular')
#         self.norm = nn.BatchNorm1d(c_in)
#         self.activation = nn.ELU()
#         self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
#
#     def forward(self, x):
#         x = self.downConv(x.permute(0, 2, 1))
#         x = self.norm(x)
#         x = self.activation(x)
#         x = self.maxPool(x)
#         x = x.transpose(1, 2)
#         return x

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            # tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
           x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
           x = self.norm(x)

        if self.projection is not None:
           x = self.projection(x)
        return x

class Decoder_change_1(nn.Module):
    def __init__(self, d_model, dropout=0.3, activation="relu", d_ff=None):
        super(Decoder_change_1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=127, out_channels=127 * 2, kernel_size=3, stride=4)
        self.conv2 = nn.Conv1d(in_channels=127 * 2, out_channels=500, kernel_size=2, stride=2)
        self.norm1 = nn.LayerNorm(107)
        self.norm2 = nn.LayerNorm(53)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear_layer_1 = nn.Linear(in_features=500 * 53, out_features=1024, bias=True)
        self.linear_layer_2 = nn.Linear(in_features=1024, out_features=500, bias=True)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):  # self.decoder_0612(enc_out)
            y = self.activation(self.conv1(x))  # (64,254,107)

            y = self.norm1(y)  #
            y = self.activation(self.conv2(y))  # (64,500,53)
            # y = self.norm2(y)
            y = y.view(y.shape[0], -1)  # y = y.flatten()
            y = self.linear_layer_1(y)
            y = self.dropout(y)
            # y = self.dropout(self.linear_layer_2(y))
            y = self.linear_layer_2(y)  # 第二层全连接不适用dropout 效果不错

            y = y.reshape(y.shape[0], 500, -1)  # y (8,500,1)
            return y

class Decoder_change(nn.Module):
        def __init__(self, layers, norm_layer=None, projection=None):
            super(Decoder_change, self).__init__()

            # Define the first 1D convolutional layer
            self.conv1 = nn.Conv1d(in_channels=127, out_channels=254, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()

            # Define the second 1D convolutional layer
            self.conv2 = nn.Conv1d(in_channels=254, out_channels=500, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()

            # Define the first fully connected layer
            # self.fc2 = nn.Linear(64, 1)
            self.li = nn.Linear(64,1)
            # self.liea = nn.Linear(500,500)

        def forward(self, x):

            # Pass through the first convolutional layer
            x = self.conv1(x)
            x = self.relu1(x)

            # Pass through the second convolutional layer
            x = self.conv2(x)
            x = self.relu2(x)

            # x = self.fc2(x)
            x = self.li(x)
            # x = self.liea(x.permute(0,2,1)).permute(0,2,1)

            return x



class Decoder_change_FB(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder_change_FB, self).__init__()

        # Define the first 1D convolutional layer
        self.conv1 = nn.Conv1d(in_channels=141, out_channels=254, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        # Define the second 1D convolutional layer
        self.conv2 = nn.Conv1d(in_channels=254, out_channels=500, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        # Define the first fully connected layer
        self.fc2 = nn.Linear(428, 1)

    def forward(self, x):
        # Pass through the first convolutional layer
        x = self.conv1(x)
        x = self.relu1(x)

        # Pass through the second convolutional layer
        x = self.conv2(x)
        x = self.relu2(x)

        x = self.fc2(x)

        return x

class Decoder_change_FB000(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder_change_FB000, self).__init__()

        # Define the first 1D convolutional layer
        self.conv1 = nn.Conv1d(in_channels=65, out_channels=130, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=130,out_channels=254,kernel_size=3,padding=1)
        self.relu2 = nn.ReLU()

        # Define the second 1D convolutional layer
        self.conv3 = nn.Conv1d(in_channels=254, out_channels=500, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        # Define the first fully connected layer
        self.fc2 = nn.Linear(428, 1)

    def forward(self, x):
        # Pass through the first convolutional layer
        x = self.conv1(x)
        x = self.relu1(x)

        # Pass through the second convolutional layer
        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.fc2(x)

        return x
class Decoder_change_FB_ICB(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder_change_FB_ICB, self).__init__()

        self.Conv1 = nn.Conv1d(428, 856, 1)
        self.Conv2 = nn.Conv1d(428, 856, 3, 1, padding=1)
        self.Conv3 = nn.Conv1d(856, 428, 1)
        self.drop = nn.Dropout(0)
        self.act = nn.GELU()

        # Define the first 1D convolutional layer
        self.conv1 = nn.Conv1d(in_channels=141, out_channels=254, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        # Define the second 1D convolutional layer
        self.conv2 = nn.Conv1d(in_channels=254, out_channels=500, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        # Define the first fully connected layer
        self.fc2 = nn.Linear(428, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.Conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.Conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.Conv3(out1 + out2)
        x = x.transpose(1, 2)

        # Pass through the first convolutional layer
        x = self.conv1(x)
        x = self.relu1(x)

        # Pass through the second convolutional layer
        x = self.conv2(x)
        x = self.relu2(x)

        x = self.fc2(x)

        return x

class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )

        out = out.view(B, L, -1)
        return self.out_projection(out), attn
