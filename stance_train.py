# built-in module
import os
import pickle
import random

# 3rd-party module
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# self-made module
import util

# prevent warning
pd.options.mode.chained_assignment = None

def main():
    # get config from command line
    config = util.config.parse_args()

    # define save path
    save_path = f'model/{config.experiment_no}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        raise FileExistsError(f'experiment {config.experiment_no} have already exist')

    # save config
    util.config.save(config, f'{save_path}/config.json')

    # initialize device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    # set random seed and ensure deterministic
    os.environ['PYTHONHASHSEED'] = str(config.random_seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # load data
    data_df = util.data.load_dataset(dataset='semeval2016_train')
    train_df, valid_df = train_test_split(data_df,
                                          test_size=config.test_size,
                                          random_state=config.random_seed)

    # initialize tokenizer
    if config.tokenizer == 'BaseTokenizer':
        tokenizer = util.tokenizer.BaseTokenizer(config)
    elif config.tokenizer == 'WordPunctTokenizer':
        tokenizer = util.tokenizer.WordPunctTokenizer(config)
    elif config.tokenizer == 'TweetTokenizer':
        tokenizer = util.tokenizer.TweetTokenizer(config)

    # initialize embedding
    embedding = util.embedding.BaseEmbedding(embedding_dim=config.embedding_dim)

    # get all tokens and word embeddings
    all_sentence = []
    all_sentence.extend(train_df['target'].drop_duplicates().tolist())
    all_sentence.extend(train_df['claim'].drop_duplicates().tolist())
    all_sentence.extend(valid_df['target'].drop_duplicates().tolist())
    all_sentence.extend(valid_df['claim'].drop_duplicates().tolist())

    all_tokens = tokenizer.get_all_tokens(all_sentence)
    embedding.load_embedding(all_tokens)

    # build vocabulary dictionary
    tokenizer.build_dict(embedding.word_dict)

    # content encode
    train_df['target_encode'] = \
        tokenizer.encode(train_df['target'].tolist(), padding=False)
    train_df['claim_encode'] = \
        tokenizer.encode(train_df['claim'].tolist(), padding=False)
    valid_df['target_encode'] = \
        tokenizer.encode(valid_df['target'].tolist(), padding=False)
    valid_df['claim_encode'] = \
        tokenizer.encode(valid_df['claim'].tolist(), padding=False)

    # label encode
    stance_label = {'favor': 0, 'against': 1, 'none': 2}
    sentiment_label = {'pos': 0, 'neg': 1, 'other': 2}

    train_df['stance_encode'] = train_df['stance'].apply(
        lambda label: stance_label[label])
    train_df['sentiment_encode'] = train_df['sentiment'].apply(
        lambda label: sentiment_label[label])
    valid_df['stance_encode'] = valid_df['stance'].apply(
        lambda label: stance_label[label])
    valid_df['sentiment_encode'] = valid_df['sentiment'].apply(
        lambda label: sentiment_label[label])

    # load lexicon
    lexicons = util.data.load_lexicon(lexicon=config.lexicon)

    # build lexicon dictionary
    tokenizer.build_lexicon_dict(lexicons)

    # lexicon encode
    train_df['claim_lexicon'] = \
        tokenizer.encode_to_lexicon(train_df['claim_encode'].tolist())
    valid_df['claim_lexicon'] = \
        tokenizer.encode_to_lexicon(valid_df['claim_encode'].tolist())

    # save tokenizer and embedding
    tokenizer.save(f'{save_path}/tokenizer.pickle')
    embedding.save(f'{save_path}/embedding.pickle')

    # initialize tensorboard
    writer = SummaryWriter(f'tensorboard/exp-{config.experiment_no}')

    # initialize loss and f1 score
    best_train_loss, best_valid_loss = None, None
    best_train_f1, best_valid_f1 = None, None
    best_epoch = None

    # construct dataset
    train_dataset = util.data.Dataset(
        tokenizer=tokenizer,
        embedding=embedding,
        target=train_df['target'],
        target_encode=train_df['target_encode'],
        claim_encode=train_df['claim_encode'],
        claim_lexicon=train_df['claim_lexicon'],
        stance_encode=train_df['stance_encode'],
        sentiment_encode=train_df['sentiment_encode'])
    valid_dataset = util.data.Dataset(
        tokenizer=tokenizer,
        embedding=embedding,
        target=train_df['target'],
        target_encode=valid_df['target_encode'],
        claim_encode=valid_df['claim_encode'],
        claim_lexicon=valid_df['claim_lexicon'],
        stance_encode=valid_df['stance_encode'],
        sentiment_encode=valid_df['sentiment_encode'])

    # construct dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=util.data.Dataset.collate_fn)
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=util.data.Dataset.collate_fn)

    # construct model, optimizer and scheduler
    model = util.model.BaseModel(config=config,
                                 num_embeddings=embedding.get_num_embeddings(),
                                 padding_idx=tokenizer.pad_token_id,
                                 embedding_weight=embedding.vector)
    model = model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=config.learning_rate,
                                 weight_decay=config.weight_decay)

    if float(config.lr_decay) != 1:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=config.lr_decay_step,
            gamma=config.lr_decay)

    # training model
    model.zero_grad()

    for epoch in range(int(config.epoch)):
        print('\n')
        model.train()
        train_iterator = tqdm(train_dataloader, total=len(train_dataloader),
                              desc=f'epoch {epoch}', position=0)

        for _, train_target, train_claim, train_lexicon, \
            train_stance, train_sentiment in train_iterator:
            # specify device for data
            train_target = train_target.to(device)
            train_claim = train_claim.to(device)
            train_lexicon = train_lexicon.to(device)
            train_stance = train_stance.to(device)
            train_sentiment = train_sentiment.to(device)

            # get predict label and attention weight
            stance_pred, sentiment_pred, stance_weight, sentiment_weight = \
                model(train_target, train_claim)

            # clean up gradient
            optimizer.zero_grad()

            # calculate loss
            batch_loss, _ = util.loss.loss_function(
                stance_predict=stance_pred,
                stance_target=train_stance,
                sentiment_predict=sentiment_pred,
                sentiment_target=train_sentiment,
                lexicon_vector=train_lexicon,
                stance_weight=stance_weight,
                sentiment_weight=sentiment_weight,
                stance_loss_weight=config.stance_loss_weight,
                lexicon_loss_weight=config.lexicon_loss_weight)

            # backward pass
            batch_loss.backward()

            # prevent gradient boosting or vanishing
            if config.clip_grad_value != 0:
                torch.nn.utils.clip_grad_value_(model.parameters(),
                                                config.clip_grad_value)

            # gradient decent
            optimizer.step()

            # apply scheduler
            if float(config.lr_decay) != 1:
                scheduler.step()

        # evaluate model
        train_iterator = tqdm(train_dataloader, total=len(train_dataloader),
                              desc='evaluate training data', position=0)
        (train_total_loss, train_stance_loss,
         train_sentiment_loss, train_lexicon_loss,
         train_target_f1, train_macro_f1,
         train_micro_f1, train_sentiment_f1) = \
            util.evaluate.evaluate_function(device=device,
                                            model=model,
                                            config=config,
                                            batch_iterator=train_iterator)

        valid_iterator = tqdm(valid_dataloader, total=len(valid_dataloader),
                              desc='evaluate validation data', position=0)
        (valid_total_loss, valid_stance_loss,
         valid_sentiment_loss, valid_lexicon_loss,
         valid_target_f1, valid_macro_f1,
         valid_micro_f1, valid_sentiment_f1) = \
            util.evaluate.evaluate_function(device=device,
                                            model=model,
                                            config=config,
                                            batch_iterator=valid_iterator)

        # print loss and score
        print(f'train total loss: {round(train_total_loss, 5)}, '
              f'train stance loss: {round(train_stance_loss, 5)}, '
              f'train lexicon loss: {round(train_lexicon_loss, 5)}\n'
              f'train macro f1: {round(train_macro_f1, 5)}, '
              f'train micro f1: {round(train_micro_f1, 5)}\n'
              f'valid total loss: {round(valid_total_loss, 5)}, '
              f'valid stance loss: {round(valid_stance_loss, 5)}, '
              f'valid lexicon loss: {round(valid_lexicon_loss, 5)}\n'
              f'valid macro f1: {round(valid_macro_f1, 5)}, '
              f'valid micro f1: {round(valid_micro_f1, 5)}')

        if (best_valid_loss is None) or \
            (valid_stance_loss < best_valid_loss) or \
            (valid_micro_f1 > best_valid_f1):
            best_train_loss = train_total_loss
            best_train_f1 = train_micro_f1
            best_valid_loss = valid_total_loss
            best_valid_f1 = valid_micro_f1
            best_epoch = epoch

            # save model
            torch.save(model.state_dict(), f'{save_path}/model_{epoch}.ckpt')

        # write loss and f1 to tensorboard
        writer.add_scalar('Loss/train_total', train_total_loss, epoch)
        writer.add_scalar('Loss/train_stance', train_stance_loss, epoch)
        writer.add_scalar('Loss/train_sentiment', train_sentiment_loss, epoch)
        writer.add_scalar('Loss/train_lexicon', train_lexicon_loss, epoch)

        writer.add_scalar('Loss/valid_total', valid_total_loss, epoch)
        writer.add_scalar('Loss/valid_stance', valid_stance_loss, epoch)
        writer.add_scalar('Loss/valid_sentiment', valid_sentiment_loss, epoch)
        writer.add_scalar('Loss/valid_lexicon', valid_lexicon_loss, epoch)

        writer.add_scalar('F1/train_macro', train_macro_f1, epoch)
        writer.add_scalar('F1/train_micro', train_micro_f1, epoch)
        writer.add_scalar('F1/train_sentiment', train_sentiment_f1, epoch)

        writer.add_scalar('F1/valid_macro', valid_macro_f1, epoch)
        writer.add_scalar('F1/valid_micro', valid_micro_f1, epoch)
        writer.add_scalar('F1/valid_sentiment', valid_sentiment_f1, epoch)

        writer.add_scalars('F1/train_target',
                           {'atheism': train_target_f1[0],
                            'climate': train_target_f1[1],
                            'feminist': train_target_f1[2],
                            'hillary': train_target_f1[3],
                            'abortion': train_target_f1[4]}, epoch)
        writer.add_scalars('F1/valid_target',
                           {'atheism': valid_target_f1[0],
                            'climate': valid_target_f1[1],
                            'feminist': valid_target_f1[2],
                            'hillary': valid_target_f1[3],
                            'abortion': valid_target_f1[4]}, epoch)

    # print final result
    print(f'\nexperiment {config.experiment_no}: epoch {best_epoch}\n'
          f'best train total loss : {best_train_loss}, '
          f'best train stance loss: {best_valid_loss}\n'
          f'best train stance f1  : {best_train_f1}, '
          f'best valid stance f1  : {best_valid_f1}')

    # add hyperparameters and final result to tensorboard
    writer.add_hparams({
        'epoch': best_epoch,
        'train_loss': best_train_loss,
        'valid_loss': best_valid_loss,
        'train_f1': best_train_f1,
        'valid_f1': best_valid_f1
    }, metric_dict={})
    writer.add_hparams(
        {key: str(value) for key, value in config.__dict__.items()},
        metric_dict={})
    writer.close()

    # release GPU memory
    torch.cuda.empty_cache()

    return

if __name__ == '__main__':
    main()