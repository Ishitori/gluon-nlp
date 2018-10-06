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

import pytest

import mxnet as mx
from mxnet import init, nd, autograd
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader, SimpleDataset
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from types import SimpleNamespace

import gluonnlp as nlp
from gluonnlp.data import SQuAD
from scripts.question_answering.bidaf import BidirectionalAttentionFlow
from scripts.question_answering.data_processing import SQuADTransform, VocabProvider
from scripts.question_answering.performance_evaluator import PerformanceEvaluator
from scripts.question_answering.question_answering import *
from scripts.question_answering.question_id_mapper import QuestionIdMapper
from scripts.question_answering.similarity_function import DotProductSimilarity
from scripts.question_answering.tokenizer import BiDAFTokenizer
from scripts.question_answering.train_question_answering import get_record_per_answer_span

question_max_length = 30
context_max_length = 400
max_chars_per_word = 16
embedding_size = 100


@pytest.mark.serial
def test_transform_to_nd_array():
    dataset = SQuAD(segment='dev', root='tests/data/squad')
    vocab_provider = VocabProvider(dataset)
    transformer = SQuADTransform(vocab_provider, question_max_length,
                                 context_max_length, max_chars_per_word)
    record = dataset[0]

    transformed_record = transformer(*record)
    assert transformed_record is not None
    assert len(transformed_record) == 7


@pytest.mark.serial
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


@pytest.mark.serial
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
    loadable_data, dataloader = get_record_per_answer_span(processed_dataset, get_args(batch_size))

    word_vocab = vocab_provider.get_word_level_vocab()
    word_vocab.set_embedding(nlp.embedding.create('glove', source='glove.6B.100d'))
    char_vocab = vocab_provider.get_char_level_vocab()

    embedding = BiDAFEmbedding(word_vocab=word_vocab,
                               char_vocab=char_vocab,
                               batch_size=batch_size,
                               max_seq_len=question_max_length,
                               precision="float16")
    embedding.cast("float16")
    embedding.initialize(init.Xavier(magnitude=2.24))
    embedding.hybridize(static_alloc=True)
    state = embedding.begin_state()

    trainer = Trainer(embedding.collect_params(), "sgd", {"learning_rate": 0.1,
                                                          "multi_precision": True})

    for i, (data, label) in enumerate(dataloader):
        with autograd.record():
            record_index, q_words, ctx_words, q_chars, ctx_chars = data
            q_words = q_words.astype("float16")
            ctx_words = ctx_words.astype("float16")
            q_chars = q_chars.astype("float16")
            ctx_chars = ctx_chars.astype("float16")
            label = label.astype("float16")
            # passing only question_words_nd and question_chars_nd batch
            out = embedding(q_words, q_chars, state)
            assert out is not None

        out.backward()
        trainer.step(batch_size)
        break


def test_attention_layer():
    batch_size = 5

    ctx_fake_data = nd.random.uniform(shape=(batch_size, context_max_length, 2 * embedding_size),
                                      dtype="float16")

    q_fake_data = nd.random.uniform(shape=(batch_size, question_max_length, 2 * embedding_size),
                                    dtype="float16")

    ctx_fake_mask = nd.ones(shape=(batch_size, context_max_length), dtype="float16")
    q_fake_mask = nd.ones(shape=(batch_size, question_max_length), dtype="float16")

    layer = BidirectionalAttentionFlow(DotProductSimilarity(),
                                       batch_size,
                                       context_max_length,
                                       question_max_length,
                                       2 * embedding_size)

    layer.cast("float16")
    layer.initialize()
    layer.hybridize(static_alloc=True)

    with autograd.record():
        output = layer(ctx_fake_data, q_fake_data, q_fake_mask, ctx_fake_mask)

    assert output.shape == (batch_size, context_max_length, 8 * embedding_size)


def test_modeling_layer():
    batch_size = 5

    # The modeling layer receive input in a shape of batch_size x T x 8d
    # T is the sequence length of context which is context_max_length
    # d is the size of embedding, which is embedding_size
    fake_data = nd.random.uniform(shape=(batch_size, context_max_length, 8 * embedding_size),
                                  dtype="float16")
    # We assume that attention is already return data in TNC format
    attention_output = nd.transpose(fake_data, axes=(1, 0, 2))

    layer = BiDAFModelingLayer(batch_size, precision="float16")
    layer.cast("float16")
    layer.initialize()
    layer.hybridize(static_alloc=True)
    state = layer.begin_state()

    trainer = Trainer(layer.collect_params(), "sgd", {"learning_rate": "0.1",
                                                      "multi_precision": True})

    with autograd.record():
        output = layer(attention_output, state)

    output.backward()
    # According to the paper, the output should be 2d x T
    assert output.shape == (context_max_length, batch_size, 2 * embedding_size)


def test_output_layer():
    batch_size = 5

    # The output layer receive 2 inputs: the output of Modeling layer (context_max_length,
    # batch_size, 2 * embedding_size) and the output of Attention flow layer
    # (batch_size, context_max_length, 8 * embedding_size)

    # The modeling layer returns data in TNC format
    modeling_output = nd.random.uniform(shape=(context_max_length, batch_size, 2 * embedding_size),
                                        dtype="float16")
    # The layer assumes that attention is already return data in TNC format
    attention_output = nd.random.uniform(shape=(context_max_length, batch_size, 8 * embedding_size),
                                         dtype="float16")

    layer = BiDAFOutputLayer(batch_size, precision="float16")
    layer.cast("float16")
    # The model doesn't need to know the hidden states, so I don't hold variables for the states
    layer.initialize()
    layer.hybridize(static_alloc=True)
    state = layer.begin_state()

    trainer = Trainer(layer.collect_params(), "sgd", {"learning_rate": 0.1,
                                                      "multi_precision": True})

    with autograd.record():
        output = layer(attention_output, modeling_output, state)

    output.backward()
    # We expect final numbers as batch_size x 2 (first start index, second end index)
    assert output.shape == (batch_size, 2, 400)


def test_bidaf_model():
    options = get_args(batch_size=5)

    dataset = SQuAD(segment='dev', root='tests/data/squad')
    vocab_provider = VocabProvider(dataset)
    transformer = SQuADTransform(vocab_provider, question_max_length,
                                 context_max_length, max_chars_per_word)

    # for performance reason, process only batch_size # of records
    processed_dataset = SimpleDataset([transformer(*record) for i, record in enumerate(dataset)
                                       if i < options.batch_size])

    # need to remove question id before feeding the data to data loader
    loadable_data, dataloader = get_record_per_answer_span(processed_dataset, options)

    word_vocab = vocab_provider.get_word_level_vocab()
    word_vocab.set_embedding(nlp.embedding.create('glove', source='glove.6B.100d'))
    char_vocab = vocab_provider.get_char_level_vocab()

    model = BiDAFModel(word_vocab=word_vocab,
                       char_vocab=char_vocab,
                       options=options)

    model.cast("float16")
    model.initialize(init.Xavier(magnitude=2.24))
    model.hybridize(static_alloc=True)

    ctx_embedding_begin_state = model.ctx_embedding.begin_state()
    q_embedding_begin_state = model.q_embedding.begin_state()
    m_layer_begin_state = model.modeling_layer.begin_state()
    o_layer_begin_state = model.output_layer.begin_state()

    loss_function = SoftmaxCrossEntropyLoss()
    trainer = Trainer(model.collect_params(), "adadelta", {"learning_rate": 0.5,
                                                           "multi_precision": True})

    for i, (data, label) in enumerate(dataloader):
        record_index, q_words, ctx_words, q_chars, ctx_chars = data
        q_words = q_words.astype("float16")
        ctx_words = ctx_words.astype("float16")
        q_chars = q_chars.astype("float16")
        ctx_chars = ctx_chars.astype("float16")
        label = label.astype("float16")

        with autograd.record():
            out = model(record_index, q_words, ctx_words, q_chars, ctx_chars,
                        ctx_embedding_begin_state, q_embedding_begin_state,
                        m_layer_begin_state, o_layer_begin_state)
            loss = loss_function(out, label)

        loss.backward()
        trainer.step(options.batch_size)
        break

    nd.waitall()


def test_performance_evaluation():
    options = get_args(batch_size=5)

    train_dataset = SQuAD(segment='train')
    vocab_provider = VocabProvider(train_dataset)

    dataset = SQuAD(segment='dev')
    mapper = QuestionIdMapper(dataset)

    transformer = SQuADTransform(vocab_provider, question_max_length,
                                 context_max_length, max_chars_per_word)

    # for performance reason, process only batch_size # of records
    transformed_dataset = SimpleDataset([transformer(*record) for i, record in enumerate(dataset)
                                         if i < options.batch_size])

    word_vocab = vocab_provider.get_word_level_vocab()
    word_vocab.set_embedding(nlp.embedding.create('glove', source='glove.6B.100d'))
    char_vocab = vocab_provider.get_char_level_vocab()
    model_path = os.path.join(options.save_dir, 'epoch{:d}.params'.format(int(options.epochs) - 1))

    ctx = [mx.cpu()]
    evaluator = PerformanceEvaluator(transformed_dataset, dataset._read_data(), mapper)
    net = BiDAFModel(word_vocab, char_vocab, options, prefix="bidaf")
    net.hybridize(static_alloc=True)
    net.load_parameters(model_path, ctx=ctx)

    result = evaluator.evaluate_performance(net, ctx, options)
    print("Evaluation results on dev dataset: {}".format(result))


# def test_count_num_of_answer_index_greater_400():
#     counter_more_400 = 0
#     counter_less_400 = 0
#     train_dataset = SQuAD(segment='train')
#
#     for item in train_dataset:
#         for index in item[5]:
#             if index >= 400:
#                 counter_more_400 += 1
#             else:
#                 counter_less_400 += 1
#
#     print("Less {}, More {}".format(counter_less_400, counter_more_400))


def test_get_answer_spans_exact_match():
    tokenizer = BiDAFTokenizer()

    context = "to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary."
    context_tokens = tokenizer(context)

    answer_start_index = 3
    answer = "Saint Bernadette Soubirous"

    result = SQuADTransform._get_answer_spans(context, context_tokens,
                                              [answer], [answer_start_index])

    assert result == [(1, 3)]


def test_get_answer_spans_partial_match():
    tokenizer = BiDAFTokenizer()

    context = "In addition, trucks will be allowed to enter India's capital only after 11 p.m., two hours later than the existing restriction"
    context_tokens = tokenizer(context)

    answer_start_index = 72
    answer = "11 p.m"

    result = SQuADTransform._get_answer_spans(context, context_tokens,
                                              [answer], [answer_start_index])

    assert result == [(16, 17)]


def test_get_answer_spans_unicode():
    tokenizer = BiDAFTokenizer()

    context = "Back in Warsaw that year, Chopin heard Niccolò Paganini play"
    context_tokens = tokenizer(context)

    answer_start_index = 39
    answer = "Niccolò Paganini"

    result = SQuADTransform._get_answer_spans(context, context_tokens,
                                              [answer], [answer_start_index])

    assert result == [(8, 9)]


def test_get_answer_spans_after_comma():
    tokenizer = BiDAFTokenizer()

    context = "Chopin's successes as a composer and performer opened the door to western Europe for him, and on 2 November 1830, he set out,"
    context_tokens = tokenizer(context)

    answer_start_index = 108
    answer = "1830"

    result = SQuADTransform._get_answer_spans(context, context_tokens,
                                              [answer], [answer_start_index])

    assert result == [(23, 23)]

def test_get_char_indices():
    context = "to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary."
    tokenizer = BiDAFTokenizer()
    context_tokens = tokenizer(context)

    result = SQuADTransform._get_char_indices(context, context_tokens)
    assert len(result) == len(context_tokens)

def get_args(batch_size):
    options = SimpleNamespace()
    options.ctx_embedding_num_layers = 2
    options.embedding_size = 100
    options.dropout = 0.2
    options.ctx_embedding_num_layers = 2
    options.highway_num_layers = 2
    options.modeling_num_layers = 2
    options.output_num_layers = 2
    options.batch_size = batch_size
    options.ctx_max_len = context_max_length
    options.q_max_len = question_max_length
    options.word_max_len = max_chars_per_word
    options.precision = "float16"
    options.epochs = 100
    options.save_dir = "output/"

    return options
