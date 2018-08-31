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
import numpy as np

from mxnet import nd
from mxnet.gluon.data import SimpleDataset

from gluonnlp import Vocab, data
from gluonnlp.data.batchify import Pad


def preprocess_dataset(dataset, question_max_length, context_max_length, max_chars_per_word):
    """Process SQuAD dataset by creating NDArray version of data

    :param Dataset dataset: SQuAD dataset
    :param int question_max_length: Maximum length of question (padded or trimmed to that size)
    :param int context_max_length: Maximum length of context (padded or trimmed to that size)
    :param int max_chars_per_word: Maximum length of word (padded or trimmed to that size)

    Returns
    -------
    SimpleDataset
        Dataset of preprocessed records
    """
    vocab_provider = VocabProvider(dataset)
    transformer = SQuADTransform(vocab_provider, question_max_length,
                                 context_max_length, max_chars_per_word)
    processed_dataset = SimpleDataset(dataset.trasform(transformer, lazy=False))
    return processed_dataset


class SQuADTransform(object):
    """SQuADTransform class responsible for converting text data into NDArrays that can be later
    feed into DataProvider
    """
    def __init__(self, vocab_provider, question_max_length, context_max_length, max_chars_per_word):
        self._word_vocab = vocab_provider.get_word_level_vocab()
        self._char_vocab = vocab_provider.get_char_level_vocab()

        self._question_max_length = question_max_length
        self._context_max_length = context_max_length
        self._max_chars_per_word = max_chars_per_word

        self._padder = Pad()

    def __call__(self, record_index, question_id, question, context, answer_list,
                 answer_start_list):
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

        question_words_nd = self._pad_to_max_word_length(question_words, self._question_max_length)
        question_chars_nd = self._padder(question_chars)
        question_chars_nd = self._pad_to_max_char_length(question_chars_nd,
                                                         self._question_max_length)

        context_words_nd = self._pad_to_max_word_length(context_words, self._context_max_length)
        context_chars_nd = self._padder(context_chars)
        context_chars_nd = self._pad_to_max_char_length(context_chars_nd, self._context_max_length)

        answer_spans = SQuADTransform._get_answer_spans(answer_list, answer_start_list)

        return (record_index, question_id, question_words_nd, context_words_nd,
                question_chars_nd, context_chars_nd, answer_spans)

    @staticmethod
    def _get_answer_spans(answer_list, answer_start_list):
        """Find all answer spans from the context, returning start_index and end_index

        :param list[str] answer_list: List of all answers
        :param list[int] answer_start_list: List of all answers' start indices

        Returns
        -------
        List[Tuple]
            list of Tuple(answer_start_index answer_end_index) per question
        """
        return [(answer_start_list[i], answer_start_list[i] + len(answer))
                for i, answer in enumerate(answer_list)]

    def _pad_to_max_char_length(self, item, max_item_length):
        """Pads all tokens to maximum size

        :param NDArray item: matrix of indices
        :param int max_item_length: maximum length of a token
        :return:
        """
        # expand dimensions to 4 and turn to float32, because nd.pad can work only with 4 dims
        data_expanded = item.reshape(1, 1, item.shape[0], item.shape[1]).astype(np.float32)
        data_padded = nd.pad(data_expanded,
                             mode='constant',
                             pad_width=[0, 0, 0, 0, 0, max_item_length - item.shape[0],
                                        0, self._max_chars_per_word - item.shape[1]],
                             constant_value=0)

        # reshape back to original dimensions with the last dimension of max_item_length
        # We also convert to float32 because it will be necessary later for processing
        data_reshaped_back = data_padded.reshape(max_item_length,
                                                 self._max_chars_per_word).astype(np.float32)
        return data_reshaped_back

    @staticmethod
    def _pad_to_max_word_length(item, max_length):
        """Pads sentences to maximum length

        :param NDArray item: vector of words
        :param int max_length: Maximum length of question/context
        :return:
        """
        data_nd = nd.array(item, dtype=np.float32)
        # expand dimensions to 4 and turn to float32, because nd.pad can work only with 4 dims
        data_expanded = data_nd.reshape(1, 1, 1, data_nd.shape[0])
        data_padded = nd.pad(data_expanded,
                             mode='constant',
                             pad_width=[0, 0, 0, 0, 0, 0, 0, max_length - data_nd.shape[0]],
                             constant_value=0)
        # reshape back to original dimensions with the last dimension of max_length
        # We also convert to float32 because it will be necessary later for processing
        data_reshaped_back = data_padded.reshape(max_length).astype(np.float32)
        return data_reshaped_back


class VocabProvider(object):
    """Provides word level and character level vocabularies
    """
    def __init__(self, dataset):
        self._dataset = dataset

    def get_char_level_vocab(self):
        """Provides character level vocabulary

        Returns
        -------
        Vocab
            Character level vocabulary
        """
        return VocabProvider._create_squad_vocab(iter, self._dataset)

    def get_word_level_vocab(self):
        """Provides word level vocabulary

        Returns
        -------
        Vocab
            Word level vocabulary
        """

        def simple_tokenize(source_str, token_delim=' ', seq_delim='\n'):
            return list(filter(None, re.split(token_delim + '|' + seq_delim, source_str)))

        return VocabProvider._create_squad_vocab(simple_tokenize, self._dataset)

    @staticmethod
    def _create_squad_vocab(tokenization_fn, dataset):
        all_tokens = []

        for data_item in dataset:
            all_tokens.extend(tokenization_fn(data_item[2]))
            all_tokens.extend(tokenization_fn(data_item[3]))

        counter = data.count_tokens(all_tokens)
        vocab = Vocab(counter)
        return vocab
