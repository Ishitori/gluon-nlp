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
import pickle

from os.path import isfile

import gluonnlp as nlp
from scripts.question_answering.tokenizer import BiDAFTokenizer

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
    def __init__(self, vocab_provider, question_max_length, context_max_length,
                 max_chars_per_word, embedding_size):
        self._word_vocab = vocab_provider.get_word_level_vocab(embedding_size)
        self._char_vocab = vocab_provider.get_char_level_vocab()
        self._tokenizer = vocab_provider.get_tokenizer()

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
        question_tokens = self._tokenizer(question)
        context_tokens = self._tokenizer(context)

        question_words = self._word_vocab[question_tokens[:self._question_max_length]]
        context_words = self._word_vocab[context_tokens[:self._context_max_length]]

        question_chars = [self._char_vocab[list(iter(word))]
                          for word in question_tokens[:self._question_max_length]]

        context_chars = [self._char_vocab[list(iter(word))]
                         for word in context_tokens[:self._context_max_length]]

        question_words_nd = self._pad_to_max_word_length(question_words, self._question_max_length)
        question_chars_nd = self._padder(question_chars)
        question_chars_nd = self._pad_to_max_char_length(question_chars_nd,
                                                         self._question_max_length)

        context_words_nd = self._pad_to_max_word_length(context_words, self._context_max_length)
        context_chars_nd = self._padder(context_chars)
        context_chars_nd = self._pad_to_max_char_length(context_chars_nd, self._context_max_length)

        answer_spans = SQuADTransform._get_answer_spans(context, context_tokens, answer_list,
                                                        answer_start_list)

        return (record_index, question_id, question_words_nd, context_words_nd,
                question_chars_nd, context_chars_nd, answer_spans)

    @staticmethod
    def _get_answer_spans(context, context_tokens, answer_list, answer_start_list):
        """Find all answer spans from the context, returning start_index and end_index.
        Each index is a index of a token

        :param list[str] context_tokens: Tokenized paragraph
        :param list[str] answer_list: List of all answers

        Returns
        -------
        List[Tuple]
            list of Tuple(answer_start_index answer_end_index) per question
        """
        answer_spans = []
        # SQuAD answers doesn't always match to used tokens in the context. Sometimes there is only
        # a partial match. We use the same method as used in original implementation:
        # 1. Find char index range for all tokens of context
        # 2. Foreach answer
        #   2.1 Find char index range for the answer (not tokenized)
        #   2.2 Find Context token indices which char indices contains answer char indices
        #   2.3. Return first and last token indices
        context_char_indices = SQuADTransform._get_char_indices(context, context_tokens)

        for answer_start_char_index, answer in zip(answer_start_list, answer_list):
            answer_token_indices = []
            answer_end_char_index = answer_start_char_index + len(answer)

            for context_token_index, context_char_span in enumerate(context_char_indices):
                if not (answer_end_char_index <= context_char_span[0] or
                        answer_start_char_index >= context_char_span[1]):
                    answer_token_indices.append(context_token_index)

            if len(answer_token_indices) == 0:
                print("Warning: Answer {} not found for context {}".format(answer, context))
            else:
                answer_span = (answer_token_indices[0],
                               answer_token_indices[len(answer_token_indices) - 1])
                answer_spans.append(answer_span)

        if len(answer_spans) == 0:
            print("Warning: No answers found for context {}".format(context_tokens))

        return answer_spans

    @staticmethod
    def _get_char_indices(text, text_tokens):
        """Match token with character indices

        :param str text: Text
        :param List[str] text_tokens: Tokens of the text
        :return: List of char_indexes where the order equals to token index
        """
        char_indices_per_token = []
        current_index = 0

        for token in text_tokens:
            current_index = text.find(token, current_index)
            char_indices_per_token.append((current_index, current_index + len(token)))
            current_index += len(token)

        return char_indices_per_token

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
    def __init__(self, datasets, options, tokenizer=BiDAFTokenizer()):
        self._datasets = datasets
        self._options = options
        self._tokenizer = tokenizer

    def get_tokenizer(self):
        """Provides tokenizer used to create vocab"""
        return self._tokenizer

    def get_char_level_vocab(self):
        """Provides character level vocabulary

        Returns
        -------
        Vocab
            Character level vocabulary
        """
        if self._options.char_vocab_path and isfile(self._options.char_vocab_path):
            return pickle.load(open(self._options.char_vocab_path, "rb"))

        all_chars = []
        for dataset in self._datasets:
            all_chars.extend(VocabProvider._get_all_tokens(iter, dataset))

        char_level_vocab = VocabProvider._create_squad_vocab(all_chars)

        if self._options.char_vocab_path:
            pickle.dump(char_level_vocab, open(self._options.char_vocab_path, "wb"))

        return char_level_vocab

    def get_word_level_vocab(self, embedding_size):
        """Provides word level vocabulary

        Returns
        -------
        Vocab
            Word level vocabulary
        """

        if self._options.word_vocab_path and isfile(self._options.word_vocab_path):
            return pickle.load(open(self._options.word_vocab_path, "rb"))

        all_words = []
        for dataset in self._datasets:
            all_words.extend(VocabProvider._get_all_tokens(self._tokenizer, dataset))

        word_level_vocab = VocabProvider._create_squad_vocab(all_words)
        word_level_vocab.set_embedding(
            nlp.embedding.create('glove', source='glove.6B.{}d'.format(embedding_size)))

        if self._options.word_vocab_path:
            pickle.dump(word_level_vocab, open(self._options.word_vocab_path, "wb"))

        return word_level_vocab

    @staticmethod
    def _create_squad_vocab(all_tokens):
        counter = data.count_tokens(all_tokens)
        vocab = Vocab(counter)
        return vocab

    @staticmethod
    def _get_all_tokens(tokenization_fn, dataset):
        all_tokens = []

        for data_item in dataset:
            all_tokens.extend(tokenization_fn(data_item[2]))
            all_tokens.extend(tokenization_fn(data_item[3]))

        return all_tokens
