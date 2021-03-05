# built-in module
import unicodedata
import random
import re

# 3rd-party module
import pandas as pd
from tqdm import tqdm
import torch

# self-made module
from util import tokenizer as tkn
from util import embedding as emb

def preprocessing(data):
    # encoding normalize
    data = [[unicodedata.normalize('NFKC', str(column))
             for column in row] for row in data]

    # change to lowercase
    data = [[column.lower().strip() for column in row] for row in data]

    return data

def preprocessing_tweet(text):
    # reference: https://github.com/zhouyiwei/tsd/blob/e1db26a829f8702f437accd42a998ce8e9344de1/utils.py#L5
    text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<URL>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    text = re.sub(r"[8:=;]['`\-]?[)d]+|[)d]+['`\-]?[8:=;]", "<SMILE>", text)
    text = re.sub(r"[8:=;]['`\-]?p+", "<LOLFACE>", text)
    text = re.sub(r"[8:=;]['`\-]?\(+|\)+['`\-]?[8:=;]", "<SADFACE>", text)
    text = re.sub(r"[8:=;]['`\-]?[\/|l*]", "<NEUTRALFACE>", text)
    text = re.sub(r"<3","<HEART>", text)
    text = re.sub(r"/"," / ", text)
    text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
    p = re.compile(r"#\S+")
    text = p.sub(lambda s: "<HASHTAG> "+s.group()+" <ALLCAPS>"
                 if s.group()[1:].isupper()
                 else " ".join(["<HASHTAG>"]+re.split(r"([A-Z][^A-Z]*)", s.group()[1:])), text)
    text = re.sub(r"([!?.]){2,}", r"\1 <REPEAT>", text)
    text = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <ELONG>", text)

    return text.lower()

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

    # only reserve the data which target is "Atheism"
    # df = df[df['Target'] == 'Atheism']

    # get necessary column
    data = []
    for _, row in df.iterrows():
        tweet = row['Tweet']
        # tweet = preprocessing_tweet(tweet)
        data.append([row['Target'], tweet, row['Stance'], row['Sentiment']])
        # data.append([row['Target'], row['Tweet'], row['Stance'], row['Sentiment']])

    # preprocessing
    data = preprocessing(data)

    # convert to dataframe
    data_df = convert_to_dataframe(data)

    return data_df

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

def load_lexicon_opinion():
    # file path
    file_path = ['data/opinion-lexicon-English/negative-words.txt',
                 'data/opinion-lexicon-English/positive-words.txt']

    # read data
    lexicons = []
    for file in file_path:
        with open(file, 'r', encoding='iso-8859-1') as f:
            for row in tqdm(f.readlines(),
                            desc=f'loading opinion lexicon data: {file[-18:-4]}'):
                word = row.strip()
                if len(word) > 0 and word[0] != ';':
                    lexicons.append(word)

    lexicons = list(set(lexicons))

    return lexicons

def load_lexicon(lexicon=None):
    # load lexicon by passed parameter
    if lexicon == 'emolex_emotion':
        return load_lexicon_emolex(types='emotion')
    elif lexicon == 'emolex_sentiment':
        return load_lexicon_emolex(types='sentiment')
    elif lexicon == 'opinion_lexicon':
        return load_lexicon_opinion()

    raise ValueError(f'lexicon {lexicon} does not support')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: tkn.BaseTokenizer,
                 embedding: emb.BaseEmbedding,
                 target, target_encode, claim_encode, claim_lexicon,
                 stance_encode, sentiment_encode):
        self.target_name = target.reset_index(drop=True)
        # self.target = target_encode
        self.target = [torch.LongTensor(ids) for ids in target_encode]
        self.claim = [torch.LongTensor(ids) for ids in claim_encode]
        self.lexicon = [torch.FloatTensor(ids) for ids in claim_lexicon]
        self.stance = torch.LongTensor([label for label in stance_encode])
        self.sentiment = torch.LongTensor([label for label in sentiment_encode])

        # get target mean embedding first
        # target_token = tokenizer.convert_ids_to_tokens(
        #     target_encode.tolist())
        # target_embeddings = [[embedding.get_embedding(token)
        #                       for token in tokens]
        #                      for tokens in target_token]
        # target_embeddings = [torch.mean(torch.Tensor(embedding), dim=0).tolist()
        #                      for embedding in target_embeddings]

        # self.target = target_embeddings
    
    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return (self.target_name[index],
                self.target[index],
                self.claim[index],
                self.lexicon[index],
                self.stance[index],
                self.sentiment[index])

    @staticmethod
    def collate_fn(batch):
        target_name = [data[0] for data in batch]
        # target = torch.FloatTensor([data[1] for data in batch])
        target = [data[1] for data in batch]
        claim = [data[2] for data in batch]
        lexicon = [data[3] for data in batch]
        stance = torch.LongTensor([data[4] for data in batch])
        sentiment = torch.LongTensor([data[5] for data in batch])

        # pad claim to fixed length with value 0
        claim = torch.nn.utils.rnn.pad_sequence(claim,
                                                batch_first=True,
                                                padding_value=0)

        # pad lexicon to fixed length with value 0.0
        lexicon = torch.nn.utils.rnn.pad_sequence(lexicon,
                                                  batch_first=True,
                                                  padding_value=0)

        return target_name, target, claim, lexicon, stance, sentiment