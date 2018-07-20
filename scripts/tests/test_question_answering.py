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
import os

from mxnet import init, nd
from mxnet.gluon.data import DataLoader, SimpleDataset

from gluonnlp.data import SQuAD
from scripts.question_answering.data_processing import SQuADTransform, VocabProvider
from scripts.question_answering.question_answering import BiDAFOutputLayer

question_max_length = 30
context_max_length = 400
embedding_size = 100


def test_transform_to_nd_array():
    dataset = SQuAD(segment='dev', root='tests/data/squad')
    vocab_provider = VocabProvider(dataset)
    transformer = SQuADTransform(vocab_provider, question_max_length, context_max_length)
    record = dataset[0]

    transformed_record = transformer(*record)
    assert transformed_record is not None
    assert len(transformed_record) == 7


def test_data_loader_able_to_read():
    dataset = SQuAD(segment='dev', root='tests/data/squad')
    vocab_provider = VocabProvider(dataset)
    transformer = SQuADTransform(vocab_provider, question_max_length, context_max_length)
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


def test_output_layer():
    batch_size = 5

    # The output layer receive 2 inputs: the output of Modeling layer (context_max_length,
    # batch_size, 2 * embedding_size) and the output of Attention flow layer
    # (batch_size, context_max_length, 8 * embedding_size)

    # The modeling layer returns data in TNC format
    modeling_output = nd.random.uniform(shape=(context_max_length, batch_size, 2 * embedding_size))
    # The layer assumes that attention is already return data in TNC format
    attention_output = nd.random.uniform(shape=(context_max_length, batch_size, 8 * embedding_size))

    layer = BiDAFOutputLayer()
    # The model doesn't need to know the hidden states, so I don't hold variables for the states
    layer.initialize()

    output = layer(attention_output, modeling_output)
    # We expect final numbers as batch_size x 2 (first start index, second end index)
    assert output.shape == (batch_size, 2)
