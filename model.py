# model.py
# see slides for better understanding of the code below

# creating Encoder part of Transformer

import torch
import torch.nn as nn
import math


'''
Encoder part 

input Embedding 
Positional Encoding
Multihead attention
layer normalization
feed forward 
And again layer normalization
'''
# Define the input embedding layer


class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        # d_model = dimension of the model
        # vocab_size = size of the vocabulary

        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    '''
    From section 3.4 of paper , 
    
    In the embedding layers, we multiply those weights by √dmodel.
    '''

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

    # now we have done Embedding so move on to Positional Embedding


# Define the Positional Embedding layer
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        # seq_len = length of the sequence
        # d_model = dimension of the model

        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

    # create a matrix of shape (seq_len,d_model)

        pe = torch.zeros(seq_len, d_model)

    # create a vectore of shape(seq_len, 1)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float(
        ) * (-math.log(10000.0)/d_model))  # for sin n cos terms
        # Apply the sin to even Position
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the cos to odd Position
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # buffer variable is used to store the positional embedding matrix because it is not trainable
        self.register_buffer('pe', pe)

    def forward(self, x):

        # x = (batch_size, seq_len, d_model)
        # requires_grad_(False) is used to make the tensor non-trainable
        # this is done because we don't want to update the positional embedding matrix during training
        # so we need to make it non-trainable
        # here we are adding the positional embedding matrix to the input tensor

        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        x = self.dropout(x)
        return x


# Now layer normlization class
class LayerNormalization(nn.Module):
    # eps = epsilon ,which is usedf for numerical stability and also to avoid divison by zero
    def __init__(self, d_model, eps=1e-6):

        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))  # multiplied aplha
        self.beta = nn.Parameter(torch.zeros(1))  # added bias

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.gamma * (x - mean) / (std + self.eps) + self.beta


'''
section 3.3 from paper descrobe Feed Forward Network
'''


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2

        # here W1 and W2  and  B1 are learnable parameters
    def forward(self, x):
        # (Batch , seq_len , d_model_ --> (Batch , seq_len , d_ff) --> (Batch , seq_len , d_model)

        return self.linear_2(self.dropou(torch.relu(self.linear_1(x))))


'''
Now main part

Multi head Attention 
'''


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
    # H here is the number of heads
    # see PPT for more references and understanding

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv
        self.w_o = nn.Linear(d_model, d_model)  # Wo
        self.dropout = nn.Dropout(dropout)

    # mask here is for future values in seq X seq matrix after diagonal for making causal system
    @staticmethod
    def attetion(query, key, value, mask, dropout: nn.Dropout):

        d_k = query.shape[-1]

        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        attention_score = attention_score.softmax(
            dim=-1)  # (Batch., h, seq_len, seq_len)

        if dropout is not None:
            attention_score = dropout(attention_score)

        return attention_score @ value, attention_score

    def forward(self, q, k, v, mask):

        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, h, d_k)

        # now we have divided the d_model into h parts
        query = query.view(
            query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1],
                       self.h, self.d_k).transpose(1, 2)
        value = value.view(
            value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_score = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout)

        # (batch_size,h,seq_len, d_k) --> (batch_size, seq_len, h, d_k) --> (batch_size, seq_len, d_model)

        x = x.transpose(1, 2).contiguous().view(
            x.shape[0], x.shape[1], self.d_model)

        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        return self.w_o(x)


'''
Now encoder decoder connection

keys and values 

and as we know ,query will be received from mask multihead attention

'''


class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

    '''
    iN ppT N PAPER THE nX IN IMAGE REPRESET n 
    TIMES ENCODEER SO EXCPET THE LAST EVERY OUTPUT OF ENCODER GOES BACK TO ANOTHER ENCODER AND LAST ONE'S OUTPUT GOES IN DECCODER
    '''


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)])

     # x is input of decoder ,encoder output ,src_mask is for one language(encoder) and tgt_mask is for other language(decoder)
    def forward(self, x, encoder_oputput, src_mask, tgt_mask):

        x = self.residual_connections[0](
            x, lambda x: self.self_attaentionm_block(x, x, x, tgt_mask))  # masked self attention and thats why decoder mask

        # cross attention for encoder's key(encoder_oputput)  and value(encoder_oputputand with them encoder mask ,src_mask
        x = self.residual_connections[1](
            x, lambda x: self.cross_attention_block(x, encoder_oputput, encoder_oputput, src_mask))

        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


'''
Encodder DEcoder black is comapleted  Now its time to code Linear and softmax 

and then after Connection of encoder and decoder

'''

# Linear step is basically position the output embedding into vocabluary


class ProjectionLayer (nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch , seq_len , d_model) --> (Batch , seq_len , vocab_size)
        # log softmax to avoid overflow and stability
        return F.log_softmax(self.linear(x), dim=-1)


'''
Now its time to code the whole model 

Tranformer model
'''


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, PrjectionLayer: ProjectionLayer) -> None:

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj = ProjectionLayer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_oputput, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_oputput, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(
            d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(
            d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(
            d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block,
                                     decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
