# 3rd-party module
import torch
from torch import nn
from torch.nn import functional as F

class BaseModel(torch.nn.Module):
    def __init__(self, device, config, num_embeddings,
                 padding_idx, embedding_weight=None):
        super(BaseModel, self).__init__()

        # device
        self.device = device

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

        self.rnn_dp = nn.Dropout(p=config.rnn_dropout)

    def forward(self, batch_target, batch_claim):
        # get target embedding
        batch_target = [[self.embedding_layer.weight[id] for id in ids]
                        for ids in batch_target]

        # get average of target embedding
        temp = []
        for target in batch_target:
            sum_embedding = 0.0
            for embedding in target:
                sum_embedding += embedding
            temp.append((sum_embedding/len(target)).unsqueeze(0))
        
        batch_target = torch.cat(temp, dim=0).to(self.device)

        # embedding layer
        batch_claim = self.rnn_dp(self.embedding_layer(batch_claim))

        # stance attn
        stance_r, stance_weight = self.stance_attn(batch_target, batch_claim)

        # sentiment attn
        sentiment_r, sentiment_weight = self.sentiment_attn(batch_claim)

        # stance linear and softmax
        stance_r = self.stance_linear(stance_r, sentiment_r)

        # sentiment linear and softmax
        sentiment_r = self.sentiment_linear(sentiment_r)

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

        # linear layer for v_s^T
        self.v_linear = nn.Linear(in_features=2*config.hidden_dim,
                                  out_features=1,
                                  bias=False)

    def forward(self, batch_target, batch_claim):
        # get all hidden vector
        claim_ht, _ = self.LSTM(batch_claim)  # (B, S, H)

        # get attention vector e
        e = torch.tanh(self.t_linear(batch_target).unsqueeze(1) + # (B, 1, H)
                       self.h_linear(claim_ht))  # (B, S, H)

        e = self.v_linear(e).squeeze(dim=2)  # (B, S)

        # apply softmax to get attention score
        weight = torch.nn.functional.softmax(e, dim=1)  # (B, S)

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

        # linear layer for v_t^T
        self.v_linear = nn.Linear(in_features=2*config.hidden_dim,
                                  out_features=1,
                                  bias=False)

    def forward(self, batch_claim):
        # get all hidden vector
        claim_ht, _ = self.LSTM(batch_claim)  # (B, S, H)

        # get final hidden vector
        final_claim_ht = claim_ht[:, -1]  # (B, H)

        # get attention vector e
        e = torch.tanh(self.s_linear(final_claim_ht).unsqueeze(1) + # (B, 1, H)
                       self.h_linear(claim_ht))  # (B, S, H)

        e = self.v_linear(e).squeeze(dim=2)  # (B, S)

        # apply softmax to get attention score
        weight = torch.nn.functional.softmax(e, dim=1)  # (B, S)

        # get final vector representation
        r = torch.matmul(weight.unsqueeze(1), claim_ht).squeeze(1)  # (B, H)

        return r, weight

class StanceLinear(torch.nn.Module):
    def __init__(self, config):
        super(StanceLinear, self).__init__()

        self.linear = nn.Sequential(
            # nn.Dropout(p=config.linear_dropout),
            nn.Linear(in_features=4*config.hidden_dim,
                      out_features=config.stance_linear_dim),
            nn.ReLU(),
            nn.Dropout(p=config.linear_dropout),
            nn.Linear(in_features=config.stance_linear_dim,
                      out_features=config.stance_output_dim),
        )

    def forward(self, stance_r, sentiment_r):
        stance_r = torch.cat((sentiment_r, stance_r), dim=1)
        stance_r = self.linear(stance_r)

        return stance_r

class SentimentLinear(torch.nn.Module):
    def __init__(self, config):
        super(SentimentLinear, self).__init__()
        self.linear = nn.Sequential(
            # nn.Dropout(p=config.linear_dropout),
            nn.Linear(in_features=2*config.hidden_dim,
                        out_features=config.sentiment_linear_dim),
            nn.ReLU(),
            nn.Dropout(p=config.linear_dropout),
            nn.Linear(in_features=config.sentiment_linear_dim,
                        out_features=config.sentiment_output_dim),
        )

    def forward(self, sentiment_r):
        sentiment_r = self.linear(sentiment_r)

        return sentiment_r