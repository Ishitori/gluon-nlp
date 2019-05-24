import itertools
import json
import multiprocessing as mp
import os
import pickle
import time

import numpy as np
import spacy
import tqdm
from mxnet.gluon.data import Dataset

import gluonnlp as nlp
from gluonnlp import data, Vocab
from gluonnlp.data import SQuAD
from scripts.question_answering.utils import MapReduce


class SQuADDataPipeline:
    def __init__(self, train_para_limit, train_ques_limit, dev_para_limit, dev_ques_limit,
                 ans_limit, char_limit, emb_size, data_root_path='./data', save_load_data=False):
        self._train_para_limit = train_para_limit
        self._train_ques_limit = train_ques_limit
        self._dev_para_limit = dev_para_limit
        self._dev_ques_limit = dev_ques_limit
        self._ans_limit = ans_limit
        self._char_limit = char_limit
        self._emb_size = emb_size
        self._data_root_path = data_root_path
        self._save_load_data = save_load_data

        self._processed_train_data_file_name = 'train_processed.json'
        self._processed_dev_data_file_name = 'dev_processed.json'
        self._word_vocab_file_name = 'word_vocab.bin'
        self._char_vocab_file_name = 'char_vocab.bin'

    def get_processed_data(self):
        if self._save_load_data and self._has_processed_data():
            return self._load_processed_data()

        train_dataset = SQuAD(segment='train')
        dev_dataset = SQuAD(segment='dev')

        with mp.Pool() as pool:
            train_examples, dev_examples = self._tokenize_data(train_dataset, dev_dataset, pool)
            word_vocab, char_vocab = self._get_vocabs(train_examples, dev_examples, pool)

        filter_provider = SQuADDataFilter(self._train_para_limit,
                                          self._train_ques_limit,
                                          self._ans_limit)
        train_examples = list(filter(filter_provider.filter, train_examples))

        train_featurizer = SQuADDataFeaturizer(word_vocab,
                                               char_vocab,
                                               self._train_para_limit,
                                               self._train_ques_limit,
                                               self._char_limit)

        dev_featuarizer = SQuADDataFeaturizer(word_vocab,
                                              char_vocab,
                                              self._dev_para_limit,
                                              self._dev_ques_limit,
                                              self._char_limit)

        train_examples, dev_examples = self._featurize_data(train_examples, dev_examples,
                                                            train_featurizer, dev_featuarizer)

        if self._save_load_data:
            self._save_processed_data(train_examples, dev_examples, word_vocab, char_vocab)

        return train_dataset._read_data(), dev_dataset._read_data(), \
               SQuADQADataset(train_examples), SQuADQADataset(dev_examples), \
               word_vocab, char_vocab

    def _tokenize_data(self, train_dataset, dev_dataset, pool):
        tokenizer = SQuADDataTokenizer()

        tic = time.time()
        print("Train examples [{}] transformation started.".format(len(train_dataset)))
        train_examples = list(tqdm.tqdm(
            pool.imap(tokenizer.tokenize_one_example, train_dataset),
            total=len(train_dataset)))
        print("Train examples transformed [{}/{}] in {:.3f} sec".format(len(train_examples),
                                                                        len(train_dataset),
                                                                        time.time() - tic))
        tic = time.time()
        print("Dev examples [{}] transformation started.".format(len(dev_dataset)))
        dev_examples = list(tqdm.tqdm(
            pool.imap(tokenizer.tokenize_one_example, dev_dataset),
            total=len(dev_dataset)))
        print("Dev examples transformed [{}/{}] in {:.3f} sec".format(len(dev_examples),
                                                                      len(dev_dataset),
                                                                      time.time() - tic))
        return train_examples, dev_examples

    def _featurize_data(self, train_examples, dev_examples, train_featurizer, dev_featuarizer):
        tic = time.time()
        print("Train examples [{}] featurization started.".format(len(train_examples)))
        train_ready = [train_featurizer.build_features(example)
                       for example in tqdm.tqdm(train_examples, total=len(train_examples))]
        print("Train examples featurized [{}] in {:.3f} sec".format(len(train_examples),
                                                                    time.time() - tic))
        tic = time.time()
        print("Dev examples [{}] featurization started.".format(len(dev_examples)))
        dev_ready = [dev_featuarizer.build_features(example)
                     for example in tqdm.tqdm(dev_examples, total=len(dev_examples))]
        print("Dev examples featurized [{}] in {:.3f} sec".format(len(dev_examples),
                                                                  time.time() - tic))
        return train_ready, dev_ready

    def _get_vocabs(self, train_examples, dev_examples, pool):
        tic = time.time()
        print("Word counters receiving started.")
        mapper = MapReduce(SQuADDataPipeline._split_into_words, SQuADDataPipeline._count_tokens)
        word_counts = mapper(itertools.chain(train_examples, dev_examples), pool)
        print("Word counters received in {:.3f} sec".format(time.time() - tic))

        tic = time.time()
        print("Char counters receiving started.")
        mapper = MapReduce(SQuADDataPipeline._split_into_chars, SQuADDataPipeline._count_tokens)
        char_counts = mapper(itertools.chain(train_examples, dev_examples), pool)
        print("Char counters received in {:.3f} sec".format(time.time() - tic))

        word_vocab = Vocab({item[0]: item[1] for item in word_counts},
                           bos_token=None, eos_token=None)
        word_vocab.set_embedding(nlp.embedding.create('glove',
                                                      source='glove.6B.{}d'.format(300)))
        char_vocab = Vocab({item[0]: item[1] for item in char_counts},
                           bos_token=None, eos_token=None)

        # with open('./data/word_counter.txt', 'w') as f:
        #     for key, value in word_counts:
        #         f.write('{}: {}\n'.format(key, value))
        #
        # with open('./data/char_counter.txt', 'w') as f:
        #     for key, value in char_counts:
        #         f.write('{}: {}\n'.format(key, value))
        #
        # with open('./data/word_vocab.txt', 'w') as f:
        #     for token in word_vocab.idx_to_token:
        #         f.write('{}\n'.format(token))
        #
        # with open('./data/char_vocab.txt', 'w') as f:
        #     for token in char_vocab.idx_to_token:
        #         f.write('{}\n'.format(token))

        return word_vocab, char_vocab

    def _has_processed_data(self):
        return \
            os.path.exists(
                os.path.join(self._data_root_path, self._processed_train_data_file_name)) and \
            os.path.exists(
                os.path.join(self._data_root_path, self._processed_dev_data_file_name)) and \
            os.path.exists(
                os.path.join(self._data_root_path, self._word_vocab_file_name)) and \
            os.path.exists(
                os.path.join(self._data_root_path, self._char_vocab_file_name))

    def _load_processed_data(self):
        with open(os.path.join(self._data_root_path, self._processed_train_data_file_name),
                  'r') as f:
            train_examples = json.load(f)

        with open(os.path.join(self._data_root_path, self._processed_dev_data_file_name), 'r') as f:
            dev_examples = json.load(f)

        word_vocab = pickle.load(
            open(os.path.join(self._data_root_path, self._word_vocab_file_name), 'rb'))

        char_vocab = pickle.load(
            open(os.path.join(self._data_root_path, self._char_vocab_file_name), 'rb'))

        return train_examples, dev_examples, word_vocab, char_vocab

    def _save_processed_data(self, train_examples, dev_examples, word_vocab, char_vocab):
        with open(os.path.join(self._data_root_path, self._processed_train_data_file_name),
                  'w') as f:
            json.dump(train_examples, f)

        with open(os.path.join(self._data_root_path, self._processed_dev_data_file_name), 'w') as f:
            json.dump(dev_examples, f)

        pickle.dump(word_vocab,
                    open(os.path.join(self._data_root_path, self._word_vocab_file_name), 'wb'))

        pickle.dump(char_vocab,
                    open(os.path.join(self._data_root_path, self._char_vocab_file_name), 'wb'))

    @staticmethod
    def _split_into_words(example):
        para_counter = data.count_tokens(example['context_tokens'])
        ques_counter = data.count_tokens(example['ques_tokens'])
        counter = para_counter + ques_counter
        return list(counter.items())

    @staticmethod
    def _split_into_chars(example):
        para_counter = data.count_tokens([c for tkn in example['context_tokens'] for c in tkn])
        ques_counter = data.count_tokens([c for tkn in example['ques_tokens'] for c in tkn])
        counter = para_counter + ques_counter
        return list(counter.items())

    @staticmethod
    def _count_tokens(item):
        token, counts = item
        return token, sum(counts)


class SQuADDataTokenizer:
    nlp = spacy.blank('en')

    def __init__(self):
        pass

    @staticmethod
    def tokenize_one_example(example):
        r"""
            Process one article.
        """
        index, q_id, question, context, answer_list, answer_start = example

        context = context.replace('\'\'', '\" ').replace(r'``', '\" ')
        context_tokens = SQuADDataTokenizer._word_tokenize(context)
        context_chars = [list(token) for token in context_tokens]
        spans = SQuADDataTokenizer._get_token_spans(context, context_tokens)

        ques = question.replace('\'\'', '\" ').replace('``', '\" ')
        ques_tokens = SQuADDataTokenizer._word_tokenize(ques)
        ques_chars = [list(token) for token in ques_tokens]

        y1s, y2s = [], []
        answer_texts = []

        for answer_text, answer_start in zip(answer_list, answer_start):
            answer_end = answer_start + len(answer_text)
            answer_texts.append(answer_text)
            answer_span = []
            for idx, span in enumerate(spans):
                if not (answer_end <= span[0] or answer_start >= span[1]):
                    answer_span.append(idx)
            y1, y2 = answer_span[0], answer_span[-1]
            y1s.append(y1)
            y2s.append(y2)

        result = {'context_tokens': context_tokens, 'context_chars': context_chars,
                  'ques_tokens': ques_tokens, 'ques_chars': ques_chars, 'y1s': y1s,
                  'y2s': y2s, 'id': q_id, 'context': context, 'spans': spans, 'record_idx': index}
        return result

    @staticmethod
    def _word_tokenize(sent):
        r"""
        Tokenize sentence.
        """
        doc = SQuADDataTokenizer.nlp(sent)
        return [token.text for token in doc]

    @staticmethod
    def _get_token_spans(text, tokens):
        """
            convert token idx to char idx.
        """
        current = 0
        spans = []
        for token in tokens:
            current = text.find(token, current)
            if current < 0:
                print('Token {} cannot be found'.format(token))
                raise Exception()
            spans.append((current, current + len(token)))
            current += len(token)
        return spans


class SQuADDataFilter:
    def __init__(self, para_limit, ques_limit, ans_limit):
        self._para_limit = para_limit
        self._ques_limit = ques_limit
        self._ans_limit = ans_limit

    def filter(self, example):
        return len(example['context_tokens']) <= self._para_limit and \
               len(example['ques_tokens']) <= self._ques_limit and \
               (example['y2s'][0] - example['y1s'][0]) <= self._ans_limit


class SQuADDataFeaturizer:
    def __init__(self, word_vocab, char_vocab, para_limit, ques_limit, char_limit):
        self._para_limit = para_limit
        self._ques_limit = ques_limit
        self._char_limit = char_limit

        self._word_vocab = word_vocab
        self._char_vocab = char_vocab

    def _get_word(self, word):
        for token in (word, word.lower(), word.capitalize(), word.upper()):
            if token in self._word_vocab:
                return self._word_vocab[token]

        return self._word_vocab[self._word_vocab.padding_token]

    def _get_char(self, char):
        if char in self._char_vocab:
            return self._char_vocab[char]

        return self._char_vocab[self._char_vocab.padding_token]

    def build_features(self, example):
        r"""
        Generate all features.
        """
        context_idxs = np.full([self._para_limit],
                               fill_value=self._word_vocab[self._word_vocab.padding_token],
                               dtype=np.float32)

        ctx_chars_idxs = np.full([self._para_limit, self._char_limit],
                                 fill_value=self._char_vocab[self._char_vocab.padding_token],
                                 dtype=np.float32)

        ques_idxs = np.full([self._ques_limit],
                            fill_value=self._word_vocab[self._word_vocab.padding_token],
                            dtype=np.float32)

        ques_char_idxs = np.full([self._ques_limit, self._char_limit],
                                 fill_value=self._char_vocab[self._char_vocab.padding_token],
                                 dtype=np.float32)

        context_len = min(len(example['context_tokens']), self._para_limit)
        context_idxs[:context_len] = self._word_vocab[example['context_tokens'][:context_len]]

        ques_len = min(len(example['ques_tokens']), self._ques_limit)
        ques_idxs[:ques_len] = self._word_vocab[example['ques_tokens'][:ques_len]]

        for i in range(0, context_len):
            char_len = min(len(example['context_chars'][i]), self._char_limit)
            ctx_chars_idxs[i, :char_len] = self._char_vocab[example['context_chars'][i][:char_len]]

        for i in range(0, ques_len):
            char_len = min(len(example['ques_chars'][i]), self._char_limit)
            ques_char_idxs[i, :char_len] = self._char_vocab[example['ques_tokens'][i][:char_len]]

        start, end = example['y1s'][-1], example['y2s'][-1]

        record = (example['id'],
                  example['record_idx'],
                  context_idxs,
                  ques_idxs,
                  ctx_chars_idxs,
                  ques_char_idxs,
                  start,
                  end,
                  example['context'],
                  example['spans'])

        return record


class SQuADQADataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self._data = data
        self._record_idx_to_q_id = {}

        for record in data:
            self._record_idx_to_q_id[record[0]] = record[1]

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)

    def get_q_id_by_rec_idx(self, rec_idx):
        return self._record_idx_to_q_id[rec_idx]


class SQuADDataLoaderTransformer:
    def __init__(self):
        pass

    def __call__(self, id, record_idx, ctx_idxs, ques_idxs, ctx_chars_idxs, ques_char_idxs,
                 start, end, context, spans):
        return record_idx, ctx_idxs, ques_idxs, ctx_chars_idxs, ques_char_idxs, start, end
