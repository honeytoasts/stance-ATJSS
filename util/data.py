# built-in module
import unicodedata
import random

# 3rd-party module
import pandas as pd
from tqdm import tqdm
import torch

def preprocessing(data):
    # encoding normalize
    data = [[unicodedata.normalize('NFKC', str(column))
             for column in row] for row in data]

    # change to lowercase
    data = [[column.lower().strip() for column in row] for row in data]

    return data

def convert_to_dataframe(data):
    target = [row[0] for row in data]
    claim = [row[1] for row in data]
    stance = [row[2] for row in data]
    sentiment = [row[3] for row in data]

    data_df = pd.DataFrame({'target': target, 'claim': claim,
                            'stance': stance, 'sentiment': sentiment})

    return data_df

def load_dataset_semeval2016(split='train'):
    # file path
    if split == 'train':
        file_path = 'data/semeval2016/train.csv'
    elif split == 'test':
        file_path = 'data/semeval2016/test.csv'

    # read data
    df = pd.read_csv(file_path, lineterminator='\r', encoding='iso-8859-1')

    # remove data which target is "Donald Trump"
    df = df[df['Target'] != 'Donald Trump']

    # get necessary column
    data = []
    for _, row in df.iterrows():
        data.append([row['Target'], row['Tweet'], row['Stance'], row['Sentiment']])

    # preprocessing
    data = preprocessing(data)

    # convert to dataframe
    data_df = convert_to_dataframe(data)

    return data_df  # target, claim, stance

def load_dataset(dataset=None):
    # load dataset by passed parameter
    if dataset == 'semeval2016_train':
        return load_dataset_semeval2016(split='train')
    elif dataset == 'semeval2016_test':
        return load_dataset_semeval2016(split='test')

    raise ValueError(f'dataset {dataset} does not support')

def load_lexicon_emolex(types='sentiment'):
    # file path
    file_path = ('data/emolex/'
                 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')

    # read data
    lexicons = []
    with open(file_path, 'r') as f:
        for row in tqdm(f.readlines()[1:], 
                        desc=f'loading EmoLex lexicon data'):
            word, emotion, value = row.split('\t')
            if types == 'emotion':
                if emotion not in ['negative', 'positive'] and int(value) == 1:
                    lexicons.append(word.strip())
            elif types == 'sentiment':
                if emotion in ['negative', 'positive'] and int(value) == 1:
                    lexicons.append(word.strip())

    lexicons = list(set(lexicons))

    return lexicons

def load_lexicon(lexicon=None):
    # load lexicon by passed parameter
    if lexicon == 'emolex_emotion':
        return load_lexicon_emolex(types='emotion')
    elif lexicon == 'emolex_sentiment':
        return load_lexicon_emolex(types='sentiment')

    raise ValueError(f'lexicon {lexicon} does not support')

# class SingleTaskDataset(torch.utils.data.Dataset):
#     def __init__(self, task_id,
#                  target_encode, claim_encode,
#                  claim_lexicon, label_encode):
#         # 0 for stance detection and 1 for NLI
#         self.task_id = task_id
#         self.x1 = [torch.LongTensor(ids) for ids in target_encode]
#         self.x2 = [torch.LongTensor(ids) for ids in claim_encode]
#         self.lexicon = [torch.FloatTensor(ids) for ids in claim_lexicon]
#         self.y = torch.LongTensor([label for label in label_encode])

#     def __len__(self):
#         return len(self.x1)

#     def __getitem__(self, index):
#         return (self.task_id, self.x1[index], self.x2[index],
#                 self.lexicon[index], self.y[index])

#     @staticmethod
#     def collate_fn(batch, pad_token_id=0):
#         task_id = batch[0][0]
#         x1 = [data[1] for data in batch]
#         x2 = [data[2] for data in batch]
#         lexicon = [data[3] for data in batch]
#         y = torch.LongTensor([data[4] for data in batch])

#         # pad sequence to fixed length with pad_token_id
#         x1 = torch.nn.utils.rnn.pad_sequence(x1,
#                                              batch_first=True,
#                                              padding_value=pad_token_id)
#         x2 = torch.nn.utils.rnn.pad_sequence(x2,
#                                              batch_first=True,
#                                              padding_value=pad_token_id)

#         # pad lexicon to fixed length with value "0.0"
#         lexicon = torch.nn.utils.rnn.pad_sequence(lexicon,
#                                                   batch_first=True,
#                                                   padding_value=0.0)

#         return task_id, x1, x2, lexicon, y