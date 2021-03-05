# built-in modules
import argparse
import os
import json

def parse_args():
    # construct argparser
    parser = argparse.ArgumentParser(
        description='Train the AT-JSS-Lex model'
    )

    # add argument to argparser

    # experiment_no
    parser.add_argument('--experiment_no',
                        default='1',
                        type=str)

    # preprocess
    parser.add_argument('--tokenizer',
                        default='TweetTokenizer',
                        type=str)
    parser.add_argument('--filter',
                        default='punctonly',
                        type=str)
    parser.add_argument('--min_count',
                        default=1,
                        type=int)
    parser.add_argument('--max_seq_len',
                        default=20,
                        type=int)

    # dataset and lexicon
    parser.add_argument('--dataset',
                        default='semeval2016',
                        type=str)
    parser.add_argument('--lexicon',
                        default='emolex_sentiment',
                        type=str)
    parser.add_argument('--stance_output_dim',
                        default=3,
                        type=int)
    parser.add_argument('--sentiment_output_dim',
                        default=3,
                        type=int)

    # hyperparameter
    parser.add_argument('--embedding_dim',
                        default=300,
                        type=int)
    parser.add_argument('--hidden_dim',
                        default=200,
                        type=int)
    parser.add_argument('--stance_linear_dim',
                        default=100,
                        type=int)
    parser.add_argument('--sentiment_linear_dim',
                        default=50,
                        type=int)

    parser.add_argument('--num_rnn_layers',
                        default=2,
                        type=int)
    parser.add_argument('--num_linear_layers',
                        default=2,
                        type=int)
    parser.add_argument('--rnn_dropout',
                        default=0.2,
                        type=float)
    parser.add_argument('--linear_dropout',
                        default=0.5,
                        type=float)

    parser.add_argument('--learning_rate',
                        default=1e-3,
                        type=float)
    parser.add_argument('--weight_decay',
                        default=1e-2,
                        type=float)
    parser.add_argument('--clip_grad_value',
                        default=0,
                        type=float)
    parser.add_argument('--lr_decay_step',
                        default=10,
                        type=int)
    parser.add_argument('--lr_decay',
                        default=1,
                        type=float)

    parser.add_argument('--stance_loss_weight',
                        default=0.7,
                        type=float)
    parser.add_argument('--lexicon_loss_weight',
                        default=0.025,
                        type=float)
    parser.add_argument('--ignore_label',
                        default=0,
                        type=int)

    # other
    parser.add_argument('--random_seed',
                        default=7,
                        type=int)
    parser.add_argument('--test_size',
                        default=0.15,
                        type=float)
    parser.add_argument('--epoch',
                        default=50,
                        type=int)
    parser.add_argument('--batch_size',
                        default=16,
                        type=int)

    return parser.parse_args()

def save(config: argparse.Namespace, file_path=None):
    if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
    else:
        # save configs to json format
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config.__dict__, f, ensure_ascii=False)

def load(file_path):
    if file_path is None or type(file_path) != str:
        raise ValueError('argument `file_path` should be a string')
    elif not os.path.exists(file_path):
        raise FileNotFoundError('file {} does not exist'.format(file_path))

    # load config file
    with open(file_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return argparse.Namespace(**config)