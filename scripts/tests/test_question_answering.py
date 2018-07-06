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
from mxnet import init, nd
from mxnet.gluon.data import DataLoader, SimpleDataset

import gluonnlp as nlp
from gluonnlp.data import SQuAD
from scripts.question_answering.data_processing import SQuADTransform, VocabProvider
from scripts.question_answering.question_answering import BiDAFEmbedding

question_max_length = 30
context_max_length = 400
max_chars_per_word = 16


def test_transform_to_nd_array():
    dataset = SQuAD(segment='dev', root='tests/data/squad')
    vocab_provider = VocabProvider(dataset)
    transformer = SQuADTransform(vocab_provider, question_max_length,
                                 context_max_length, max_chars_per_word)
    record = dataset[0]

    transformed_record = transformer(*record)
    assert transformed_record is not None
    assert len(transformed_record) == 7


def test_data_loader_able_to_read():
    dataset = SQuAD(segment='dev', root='tests/data/squad')
    vocab_provider = VocabProvider(dataset)
    transformer = SQuADTransform(vocab_provider, question_max_length,
                                 context_max_length, max_chars_per_word)
    record = dataset[0]

    processed_dataset = SimpleDataset([transformer(*record)])
    loadable_data = SimpleDataset([(r[0], r[2], r[3], r[4], r[5], r[6]) for r in processed_dataset])
    dataloader = DataLoader(loadable_data, batch_size=1)

    for data in dataloader:
        record_index, question_words, context_words, question_chars, context_chars, answers = data

        assert record_index is not None
        assert question_words is not None
        assert context_words is not None
        assert question_chars is not None
        assert context_chars is not None
        assert answers is not None


def test_load_vocabs():
    dataset = SQuAD(segment='dev', root='tests/data/squad')
    vocab_provider = VocabProvider(dataset)

    assert vocab_provider.get_word_level_vocab() is not None
    assert vocab_provider.get_char_level_vocab() is not None


def test_bidaf_embedding():
    batch_size = 5

    dataset = SQuAD(segment='dev', root='tests/data/squad')
    vocab_provider = VocabProvider(dataset)
    transformer = SQuADTransform(vocab_provider, question_max_length,
                                 context_max_length, max_chars_per_word)

    # for performance reason, process only batch_size # of records
    processed_dataset = SimpleDataset([transformer(*record) for i, record in enumerate(dataset)
                                       if i < batch_size])

    # need to remove question id before feeding the data to data loader
    loadable_data = SimpleDataset([(r[0], r[2], r[3], r[4], r[5], r[6]) for r in processed_dataset])
    dataloader = DataLoader(loadable_data, batch_size=5)

    word_vocab = vocab_provider.get_word_level_vocab()
    word_vocab.set_embedding(nlp.embedding.create('glove', source='glove.6B.100d'))
    char_vocab = vocab_provider.get_char_level_vocab()

    embedding = BiDAFEmbedding(word_vocab=word_vocab, char_vocab=char_vocab)
    embedding.initialize(init.Xavier(magnitude=2.24))

    contextual_embedding_h0 = nd.random.uniform(shape=(4, batch_size, 100))
    contextual_embedding_c0 = nd.random.uniform(shape=(4, batch_size, 100))

    for i, data in enumerate(dataloader):
        # passing only question_words_nd and question_chars_nd batch
        out = embedding([data[1], data[3]], [contextual_embedding_h0, contextual_embedding_c0])
        assert out is not None
        break
