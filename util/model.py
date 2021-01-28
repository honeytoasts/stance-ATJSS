# 3rd-party module
import torch
from torch import nn
from torch.nn import functional as F

class BaseModel(torch.nn.Module):
    def __init__(self, config, num_embeddings,
                 padding_idx, embedding_weight=None):
        super(BaseModel, self).__init__()

        # config
        self.config = config

        # embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=num_embeddings,
                                            embedding_dim=config.embedding_dim,
                                            padding_idx=padding_idx)
        if embedding_weight is not None:
            self.embedding_layer.weight = nn.Parameter(embedding_weight)

        # attention layer for stance
        self.stance_attn = StanceAttn(config)

        # attention layer for sentiment
        self.sentiment_attn = SentimentAttn(config)

        # linear layer for stance
        self.stance_linear = StanceLinear(config)

        # linear layer for sentiment
        self.sentiment_linear = SentimentLinear(config)

    def forward(self, batch_target, batch_claim):
        # embedding layer
        batch_claim = self.embedding_layer(batch_claim)

        # stance attn
        stance_r, stance_weight = self.stance_attn(batch_target, batch_claim)

        # sentiment attn
        sentiment_r, sentiment_weight = self.sentiment_attn(batch_claim)

        # stance linear and softmax
        stance_r = self.stance_linear(stance_r, sentiment_r)
        stance_r = torch.softmax(stance_r, dim=1)

        # sentiment linear and softmax
        sentiment_r = self.sentiment_linear(sentiment_r)
        sentiment_r = torch.softmax(sentiment_r, dim=1)

        return stance_r, sentiment_r, stance_weight, sentiment_weight

class StanceAttn(torch.nn.Module):
    def __init__(self, config):
        super(StanceAttn, self).__init__()

        # config
        self.config = config

        # get parameters of LSTM
        parameter = {'input_size': config.embedding_dim,
                     'hidden_size': config.hidden_dim,
                     'num_layers': config.num_rnn_layers,
                     'batch_first': True,
                     'dropout': config.rnn_dropout,
                     'bidirectional': True}

        # bidirectional LSTM
        self.LSTM = nn.LSTM(**parameter)

        # linear layer for W_t t + b_t
        self.t_linear = nn.Linear(in_features=config.embedding_dim,
                                  out_features=2*config.hidden_dim,
                                  bias=True)

        # linear layer for W_i' h_i
        self.h_linear = nn.Linear(in_features=2*config.hidden_dim,
                                  out_features=2*config.hidden_dim,
                                  bias=False)

    def forward(self, batch_target, batch_claim):
        # get batch sequence length
        seq_len = batch_claim.shape[1]

        # get all hidden vector
        claim_ht, _ = self.LSTM(batch_claim)  # (B, S, H)

        # get final hidden vector
        final_claim_ht = claim_ht[:, -1]  # (B, H)
        final_claim_ht = final_claim_ht.repeat_interleave(
            seq_len, 0)  # (BxS, H)
        final_claim_ht = final_claim_ht.reshape(
            -1, seq_len, final_claim_ht.shape[1])  # (B, S, H)

        # get target embedding
        target_embedding = batch_target.repeat_interleave(
            seq_len, 0)  # (BxS, H)
        target_embedding = target_embedding.reshape(
            -1, seq_len, target_embedding.shape[1])  # (B, S, H)

        # get attention vector e
        e = torch.tanh(self.t_linear(target_embedding) + \
                       self.h_linear(claim_ht))  # (B, S, H)
        e = torch.matmul(final_claim_ht.unsqueeze(3).transpose(2, 3),
                         e.unsqueeze(3))  # (B, S, 1, 1)
        e = e.reshape(-1, seq_len)  # (B, S)

        # apply softmax to get attention score
        weight = torch.softmax(e, dim=1)  # (B, S)

        # get final vector representation
        r = torch.matmul(weight.unsqueeze(1), claim_ht).squeeze(1)  # (B, H)

        return r, weight

class SentimentAttn(torch.nn.Module):
    def __init__(self, config):
        super(SentimentAttn, self).__init__()

        # config
        self.config = config

        # get parameters of LSTM
        parameter = {'input_size': config.embedding_dim,
                     'hidden_size': config.hidden_dim,
                     'num_layers': config.num_rnn_layers,
                     'batch_first': True,
                     'dropout': config.rnn_dropout,
                     'bidirectional': True}

        # bidirectional LSTM
        self.LSTM = nn.LSTM(**parameter)

        # linear layer for W_s s + b_s
        self.s_linear = nn.Linear(in_features=2*config.hidden_dim,
                                  out_features=2*config.hidden_dim,
                                  bias=True)

        # linear layer for W_i h_i
        self.h_linear = nn.Linear(in_features=2*config.hidden_dim,
                                  out_features=2*config.hidden_dim,
                                  bias=False)

    def forward(self, batch_claim):
        # get batch sequence length
        seq_len = batch_claim.shape[1]

        # get all hidden vector
        claim_ht, _ = self.LSTM(batch_claim)  # (B, S, H)

        # get final hidden vector
        final_claim_ht = claim_ht[:, -1]  # (B, H)
        final_claim_ht = final_claim_ht.repeat_interleave(
            seq_len, 0)  # (BxS, H)
        final_claim_ht = final_claim_ht.reshape(
            -1, seq_len, final_claim_ht.shape[1])  # (B, S, H)

        # get attention vector e
        e = torch.tanh(self.s_linear(final_claim_ht) + \
                       self.h_linear(claim_ht))  # (B, S, H)
        e = torch.matmul(final_claim_ht.unsqueeze(3).transpose(2, 3),
                         e.unsqueeze(3))  # (B, S, 1, 1)
        e = e.reshape(-1, seq_len)  # (B, S)

        # apply softmax to get attention score
        weight = torch.softmax(e, dim=1)  # (B, S)

        # get final vector representation
        r = torch.matmul(weight.unsqueeze(1), claim_ht).squeeze(1)  # (B, H)

        return r, weight

class StanceLinear(torch.nn.Module):
    def __init__(self, config):
        super(StanceLinear, self).__init__()

        # config
        self.config = config

        # linear layer
        linear = [nn.Linear(in_features=4*config.hidden_dim,
                            out_features=config.stance_linear_dim)]

        for _ in range(config.num_linear_layers-2):
            linear.append(nn.ReLU())
            linear.append(nn.Dropout(config.linear_dropout))
            linear.append(nn.Linear(in_features=config.stance_linear_dim,
                                    out_features=config.stance_linear_dim))

        linear.append(nn.ReLU())
        linear.append(nn.Dropout(config.linear_dropout))
        linear.append(nn.Linear(in_features=config.stance_linear_dim,
                                out_features=config.stance_output_dim))

        self.linear = nn.Sequential(*linear)

    def forward(self, stance_r, sentiment_r):
        stance_r = torch.cat((sentiment_r, stance_r), dim=1)
        stance_r = self.linear(stance_r)

        return stance_r

class SentimentLinear(torch.nn.Module):
    def __init__(self, config):
        super(SentimentLinear, self).__init__()

        # config
        self.config = config

        # linear layer
        linear = [nn.Linear(in_features=2*config.hidden_dim,
                            out_features=config.sentiment_linear_dim)]

        for _ in range(config.num_linear_layers-2):
            linear.append(nn.ReLU())
            linear.append(nn.Dropout(config.linear_dropout))
            linear.append(nn.Linear(in_features=config.sentiment_linear_dim,
                                    out_features=config.sentiment_linear_dim))

        linear.append(nn.ReLU())
        linear.append(nn.Dropout(config.linear_dropout))
        linear.append(nn.Linear(in_features=config.sentiment_linear_dim,
                                out_features=config.sentiment_output_dim))

        self.linear = nn.Sequential(*linear)

    def forward(self, sentiment_r):
        sentiment_r = self.linear(sentiment_r)

        return sentiment_r