# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=
"""SQuAD data preprocessing."""
__all__ = ['SQuADTransform', 'VocabProvider', 'preprocess_dataset']

import re

from mxnet import nd
from mxnet.gluon.data import SimpleDataset

from gluonnlp import Vocab, data
from gluonnlp.data.batchify import Pad


def preprocess_dataset(dataset, question_max_length, context_max_length):
    """Process SQuAD dataset by creating NDArray version of data

    :param dataset: SQuAD dataset
    :param question_max_length: Maximum length of question (will be padded or trimmed to that size)
    :param context_max_length: Maximum length of context (will be padded or trimmed to that size)
    :return: SimpleDataset
    """
    vocab_provider = VocabProvider(dataset)
    print('Collecting vocabs from the training dataset...')
    transformer = SQuADTransform(vocab_provider, question_max_length, context_max_length)
    print('Vocabs collected. Dataset [{} records] preprocessing started...'.format(len(dataset)))
    records = []

    for i, entry in enumerate(dataset):
        _, question, context, _ = entry
        records.append(transformer(i, question, context))
        print(i)

    print('Dataset preprocessing finished.')
    return SimpleDataset(records)


class SQuADTransform(object):
    """SQuADTransform class responsible for converting text data into NDArrays that can be later
    feed into DataProvider
    """
    def __init__(self, vocab_provider, question_max_length, context_max_length):
        self._word_vocab = vocab_provider.get_word_level_vocab()
        self._char_vocab = vocab_provider.get_char_level_vocab()

        self._question_max_length = question_max_length
        self._context_max_length = context_max_length

        self._padder = Pad()

    def __call__(self, record_index, question, context):
        """
        Method converts text into numeric arrays based on Vocabulary.
        Answers are not processed, as they are not needed in input
        """
        question_words = self._word_vocab[question.split()[:self._question_max_length]]
        context_words = self._word_vocab[context.split()[:self._context_max_length]]

        question_chars = [self._char_vocab[list(iter(word))]
                          for word in question.split()[:self._question_max_length]]

        context_chars = [self._char_vocab[list(iter(word))]
                         for word in context.split()[:self._context_max_length]]

        question_words_nd = nd.array(question_words)
        question_chars_nd = self._padder(question_chars)

        context_words_nd = nd.array(context_words)
        context_chars_nd = self._padder(context_chars)

        return record_index, question_words_nd, question_chars_nd, \
               context_words_nd, context_chars_nd


class VocabProvider(object):
    """Provides word level and character level vocabularies
    """
    def __init__(self, dataset):
        self._dataset = dataset

    def get_char_level_vocab(self):
        return VocabProvider._create_squad_vocab(iter, self._dataset)

    def get_word_level_vocab(self):
        def simple_tokenize(source_str, token_delim=' ', seq_delim='\n'):
            return list(filter(None, re.split(token_delim + '|' + seq_delim, source_str)))

        return VocabProvider._create_squad_vocab(simple_tokenize, self._dataset)

    @staticmethod
    def _create_squad_vocab(tokenization_fn, dataset):
        all_tokens = []

        for data_item in dataset:
            all_tokens.extend(tokenization_fn(data_item[1]))
            all_tokens.extend(tokenization_fn(data_item[2]))

        counter = data.count_tokens(all_tokens)
        vocab = Vocab(counter)
        return vocab
