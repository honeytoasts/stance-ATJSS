# built-in module
import argparse
import os
import pickle
import random

# 3rd-party module
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# self-made module
import util

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the AT-JSS-Lex model'
    )

    # add argument to argparser
    parser.add_argument('--experiment_no',
                        default='1',
                        type=str)
    parser.add_argument('--epoch',
                        default='1',
                        type=str)

    return parser.parse_args()

def main():
    # pylint: disable=no-member

    # get experiment_no and epoch
    args = parse_args()
    experiment_no, epoch = args.experiment_no, args.epoch
    model_path = f'model/{experiment_no}'

    # load config, tokenizer, embedding
    config = util.config.load(f'{model_path}/config.json')

    if config.tokenizer == 'BaseTokenizer':
        tokenizer = util.tokenizer.BaseTokenizer(config)
    elif config.tokenizer == 'WordPunctTokenizer':
        tokenizer = util.tokenizer.WordPunctTokenizer(config)
    elif config.tokenizer == 'TweetTokenizer':
        tokenizer = util.tokenizer.TweetTokenizer(config)
    tokenizer.load(f'{model_path}/tokenizer.pickle')

    embedding = util.embedding.BaseEmbedding()
    embedding.load(f'{model_path}/embedding.pickle')

    # initialize device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    # load data
    data_df = util.data.load_dataset(dataset='semeval2016_test')

    # content encode
    data_df['target_encode'] = \
        tokenizer.encode(data_df['target'].tolist(), padding=False)
    data_df['claim_encode'] = \
        tokenizer.encode(data_df['claim'].tolist(), padding=False)

    # label encode
    stance_label = {'favor': 0, 'against': 1, 'none': 2}
    sentiment_label = {'pos': 0, 'neg': 1, 'other': 2}

    data_df['stance_encode'] = data_df['stance'].apply(
        lambda label: stance_label[label])
    data_df['sentiment_encode'] = data_df['sentiment'].apply(
        lambda label: sentiment_label[label])

    # lexicon encode
    data_df['claim_lexicon'] = \
        tokenizer.encode_to_lexicon(data_df['claim_encode'].tolist())

    # construct dataset and dataloader
    dataset = util.data.Dataset(
        tokenizer=tokenizer,
        embedding=embedding,
        target=data_df['target'],
        target_encode=data_df['target_encode'],
        claim_encode=data_df['claim_encode'],
        claim_lexicon=data_df['claim_lexicon'],
        stance_encode=data_df['stance_encode'],
        sentiment_encode=data_df['sentiment_encode'])
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=util.data.Dataset.collate_fn)

    # load model
    model = util.model.BaseModel(device=device,
                                 config=config,
                                 num_embeddings=embedding.get_num_embeddings(),
                                 padding_idx=tokenizer.pad_token_id,
                                 embedding_weight=embedding.vector)
    model.load_state_dict(
        torch.load(f'{model_path}/model_{epoch}.ckpt'))
    model = model.to(device)

    # evaluate
    batch_iterator = tqdm(dataloader, total=len(dataloader),
                          desc='evaluate test data', position=0)
    (test_total_loss, test_stance_loss,
     test_sentiment_loss, test_lexicon_loss,
     test_target_f1, test_macro_f1,
     test_micro_f1, test_sentiment_f1) = \
        util.evaluate.evaluate_function(device=device,
                                        model=model,
                                        config=config,
                                        batch_iterator=batch_iterator)

    # print loss and score
    print(f'loss: {round(test_total_loss, 5)}, '
          f'stance loss: {round(test_stance_loss, 5)}, '
          f'lexicon loss: {round(test_lexicon_loss, 5)}\n'
          f'macro f1: {round(test_macro_f1, 5)}, '
          f'micro f1: {round(test_micro_f1, 5)}\n'
          f'target f1: {test_target_f1}')

    # initialize tensorboard
    writer = SummaryWriter(f'tensorboard/exp-{config.experiment_no}')

    # write loss and f1 to tensorboard
    writer.add_scalar('Loss/test_total', test_total_loss, epoch)
    writer.add_scalar('Loss/test_stance', test_stance_loss, epoch)
    writer.add_scalar('Loss/test_sentiment', test_sentiment_loss, epoch)
    writer.add_scalar('Loss/test_lexicon', test_lexicon_loss, epoch)

    writer.add_scalar('F1/test_macro', test_macro_f1, epoch)
    writer.add_scalar('F1/test_micro', test_micro_f1, epoch)
    writer.add_scalar('F1/test_sentiment', test_sentiment_f1, epoch)

    writer.add_scalars('F1/test_target',
                        {'atheism': test_target_f1[0],
                        'climate': test_target_f1[1],
                        'feminist': test_target_f1[2],
                        'hillary': test_target_f1[3],
                        'abortion': test_target_f1[4]}, epoch)
    writer.close()

    # release GPU memory
    torch.cuda.empty_cache()

    return

if __name__ == '__main__':
    main()