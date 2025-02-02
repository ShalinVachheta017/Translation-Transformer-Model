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
class ImputEmbedding(nn.Module):

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
class PositionalEmbedding(nn.Module):

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
        self.gamma = nn.Parameter(torch.ones(1)) # multiplied aplha
        self.beta = nn.Parameter(torch.zeros(1))# added bias

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        return self.gamma * ( x - mean) / (std + self.eps) + self.beta 

'''
section 3.3 from paper descrobe Feed Forward Network
'''

class FeedForwardBlock(nn.Module):
    
    def __init__( self, d_model: int, d_ff: int, dropout: float)-> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) #W1 and B1
        self.dropout = nn.Dropout (dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #W2 and B2
        
        #here W1 and W2  and  B1 are learnable parameters
    def forward(self, x):
        # (Batch , seq_len , d_model_ --> (Batch , seq_len , d_ff) --> (Batch , seq_len , d_model)
        
        return self.linear_2(self.dropou(torch.relu(self.linear_1(x))))
        
        
'''
Now main part

Multi head Attention 
'''

class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float)-> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0 , "d_model must be divisible by h"
    
    # see PPT for more references and understanding
    
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) #Wq
        self.w_k = nn.Linear(d_model, d_model) #Wk
        self.w_v = nn.Linear(d_model, d_model) #Wv
        self.w_o = nn.Linear(d_model, d_model) #Wo
        self.dropout = nn.Dropout(dropout)
        
     
    #mask here is for future values in seqxseq matrix after diagonal for making causal system
    @staticmethod
    def attetion(query, key , value , mask,dropout : nn.Dropout):
        
        d_k = query.shape[-1]
        
        attention_score = (query @ key.transpose (-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
            
        attention_score = attention_score.softmax(dim=-1) # (Batch., h, seq_len, seq_len)
        
        if dropout is not None:
            attention_score = dropout(attention_score)
        
        return attention_score @ value , attention_score
     
    def forward(self, q, k , v , mask):
        
        
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, h, d_k)
        
        #now we have divided the d_model into h parts 
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1],
                       self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        
        x,self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # (batch_size,h,seq_len, d_k) --> (batch_size, seq_len, h, d_k) --> (batch_size, seq_len, d_model)
        
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.d_model)
        
        
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        return self.w_o(x)
        
        
        
        
'''
Now encoder decoder connection

keys and values 

and ass we know ,query will be received from mask multihead attention

'''


class ResidualConnection(nn.Module):
    
    def __init__(self, dropout: float)-> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        
        return x + self.dropout(sublayer(x))
    
    
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
