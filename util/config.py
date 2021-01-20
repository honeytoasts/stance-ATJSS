# built-in modules
import os
import json

class BaseConfig:
    def __init__(self, **kwargs):
        # experiment no
        self.experiment_no = kwargs.pop('experiment_no', 1)

        # preprocess
        self.tokenizer = kwargs.pop('tokenizer', 'TweetTokenizer')
        self.filter = kwargs.pop('filter', 'punctonly')
        self.min_count = kwargs.pop('min_count', 1)
        self.max_seq_len = kwargs.pop('max_seq_len', 20)

        # dataset and lexicon
        self.dataset = kwargs.pop('dataset', 'semeval2016')
        self.lexicon_file = kwargs.pop('lexicon_file', 'emolex_sentiment')
        self.stance_output_dim = kwargs.pop('stance_output_dim', 3)
        self.sentiment_output_dim = kwargs.pop('sentiment_output_dim', 3)

        # hyperparameter
        self.embedding_dim = kwargs.pop('embedding_dim', 300)
        self.hidden_dim = kwargs.pop('hidden_dim', 200)
        self.stance_linear_dim = kwargs.pop('stance_linear_dim', 100)
        self.sentiment_linear_dim = kwargs.pop('sentiment_linear_dim', 50)

        self.num_rnn_layers = kwargs.pop('num_rnn_layers', 2)
        self.num_linear_layers = kwargs.pop('num_linear_layers', 2)
        self.rnn_dropout = kwargs.pop('stance_dropout', 0.2)
        self.linear_dropout = kwargs.pop('sentiment_dropout', 0.5)

        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.weight_decay = kwargs.pop('weight_decay', 1e-2)
        # self.clip_grad_value = kwargs.pop('clip_grad_value', 0)
        # self.lr_decay_step = kwargs.pop('lr_decay_step', 10)
        # self.lr_decay = kwargs.pop('lr_decay', 1)

        self.stance_loss_weight = kwargs.pop('stance_loss_weight', 0.7)
        self.lexicon_loss_weight = kwargs.pop('lexicon_loss_weight', 0.025)

        # others
        self.random_seed = kwargs.pop('random_seed', 7)
        self.train_test_split = kwargs.pop('train_test_split', 0.15)
        self.epoch = kwargs.pop('epoch', 50)
        self.batch_size = kwargs.pop('batch_size', 32)

    def save(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        else:
            # save config file
            with open(file_path, 'w') as f:
                hyperparameters = {
                    'experiment_no': self.experiment_no,
                    'tokenizer': self.tokenizer,
                    'filter': self.filter,
                    'min_count': self.min_count,
                    'max_seq_len': self.max_seq_len,
                    'dataset': self.dataset,
                    'lexicon_file': self.lexicon_file,
                    'stance_output_dim': self.stance_output_dim,
                    'sentiment_output_dim': self.sentiment_output_dim,
                    'embedding_dim': self.embedding_dim,
                    'hidden_dim': self.hidden_dim,
                    'stance_linear_dim': self.stance_linear_dim,
                    'sentiment_linear_dim': self.sentiment_linear_dim,
                    'num_rnn_layers': self.num_rnn_layers,
                    'num_linear_layers': self.num_linear_layers,
                    'rnn_dropout': self.rnn_dropout,
                    'linear_dropout': self.linear_dropout,
                    'learning_rate': self.learning_rate,
                    'weight_decay': self.weight_decay,
                    # 'clip_grad_value': self.clip_grad_value,
                    # 'lr_decay_step': self.lr_decay_step,
                    # 'lr_decay': self.lr_decay,
                    'stance_loss_weight': self.stance_loss_weight,
                    'lexicon_loss_weight': self.lexicon_loss_weight,
                    'random_seed': self.random_seed,
                    'train_test_split': self.train_test_split,
                    'epoch': self.epoch,
                    'batch_size': self.batch_size
                }

                json.dump(hyperparameters, f)

    @classmethod
    def load(cls, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        elif not os.path.exists(file_path):
            raise FileNotFoundError('file {} does not exist'.format(file_path))

        # load config file
        with open(file_path, 'r', encoding='utf-8') as f:
            return cls(**json.load(f))