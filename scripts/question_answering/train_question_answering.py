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

import argparse
import numpy as np
import random
from time import time

import mxnet as mx
from mxnet import init, autograd
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader, SimpleDataset, ArrayDataset
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss

import gluonnlp as nlp
from gluonnlp.data import SQuAD

from scripts.question_answering.data_processing import VocabProvider, SQuADTransform
from scripts.question_answering.metric import f1_score, exact_match_score
from scripts.question_answering.question_answering import *
from scripts.question_answering.utils import logging_config

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)


def get_data(is_train, options):
    """Get dataset and dataloader

    Parameters
    ----------
    is_train : `bool`
        If `True`, training SQuAD dataset is loaded, if `False` valiidation dataset is loaded
    options : `Namespace`
        Data transformation arguments

    Returns
    -------
    data : Tuple
        A tuple of dataset and dataloader
    """
    dataset = SQuAD(segment='train' if is_train else 'val')
    vocab_provider = VocabProvider(dataset)
    transformer = SQuADTransform(vocab_provider, options.q_max_len,
                                 options.ctx_max_len, options.word_max_len)
    # TODO: Data processing takes too long for doing experementation
    # set it to 256 to speed up thing, but need to refactor this to maybe store processed dataset
    # and vocabs. 256 is not a random number, it is 2 * batch_size, so the last batch won't cause
    # Invalid recurrent state shape after first batch is finished
    processed_dataset = SimpleDataset([transformer(*record) for i, record in enumerate(dataset)
                                       if i < 256])

    data_no_label = []
    labels = []
    global_index = 0

    # copy records to a record per answer
    for r in processed_dataset:
        # creating a set out of answer_span will deduplicate them
        for answer_span in set(r[6]):
            # need to remove question id before feeding the data to data loader
            # And I also replace index with global_index when unrolling answers
            data_no_label.append((global_index, r[2], r[3], r[4], r[5]))
            labels.append(mx.nd.array(answer_span))
            global_index += 1

    loadable_data = ArrayDataset(data_no_label, labels)
    dataloader = DataLoader(loadable_data, batch_size=options.batch_size, shuffle=True,
                            last_batch='discard')

    return dataset, dataloader


def get_vocabs(dataset, options):
    """Get word-level and character-level vocabularies

    Parameters
    ----------
    dataset : `SQuAD`
        SQuAD dataset to build vocab from
    options : `Namespace`
        Vocab building arguments

    Returns
    -------
    data : Tuple
        A tuple of word vocabulary and character vocabulary
    """
    vocab_provider = VocabProvider(dataset)

    word_vocab = vocab_provider.get_word_level_vocab()

    word_vocab.set_embedding(
        nlp.embedding.create('glove', source='glove.6B.{}d'.format(options.embedding_size)))

    char_vocab = vocab_provider.get_char_level_vocab()
    return word_vocab, char_vocab


def get_context(options):
    """Return context list to work on

    Parameters
    ----------
    options : `Namespace`
        Training arguments

    """
    if options.gpu is None:
        ctx = mx.cpu()
        print('Use CPU')
    else:
        ctx = mx.gpu(options.gpu)

    return ctx


def run_training(net, dataloader, options):
    """Get word-level and character-level vocabularies

    Parameters
    ----------
    net : `Block`
        Network to train
    dataloader : `DataLoader`
        Initialized dataloader
    options : `Namespace`
        Training arguments

    Returns
    -------
    data : Tuple
        A tuple of word vocabulary and character vocabulary
    """
    ctx = get_context(options)

    trainer = Trainer(net.collect_params(), args.optimizer, {'learning_rate': options.lr})
    eval_metrics = mx.metric.CompositeEvalMetric(metrics=[
        mx.metric.create(lambda label, pred: f1_score(pred, label)),
        mx.metric.create(lambda label, pred: exact_match_score(pred, label))
    ])
    loss_function = SoftmaxCrossEntropyLoss()

    contextual_embedding_param_shape = (4, options.batch_size, options.embedding_size)
    ctx_initial_embedding_h0 = mx.nd.random.uniform(shape=contextual_embedding_param_shape, ctx=ctx)
    ctx_initial_embedding_c0 = mx.nd.random.uniform(shape=contextual_embedding_param_shape, ctx=ctx)
    q_initial_embedding_h0 = mx.nd.random.uniform(shape=contextual_embedding_param_shape, ctx=ctx)
    q_initial_embedding_c0 = mx.nd.random.uniform(shape=contextual_embedding_param_shape, ctx=ctx)

    ctx_embedding = [ctx_initial_embedding_h0, ctx_initial_embedding_c0]
    q_embedding = [q_initial_embedding_h0, q_initial_embedding_c0]

    train_start = time()
    avg_loss = mx.nd.zeros((1,), ctx=ctx)

    for epoch_id in range(args.epochs):
        avg_loss *= 0  # Zero average loss of each epoch
        eval_metrics.reset()  # reset metrics before each epoch

        for i, (data, label) in enumerate(dataloader):
            # start timing for the first batch of epoch
            if i == 0:
                e_start = time()

            record_index, q_words, ctx_words, q_chars, ctx_chars = data
            q_words = q_words.as_in_context(ctx)
            ctx_words = ctx_words.as_in_context(ctx)
            q_chars = q_chars.as_in_context(ctx)
            ctx_chars = ctx_chars.as_in_context(ctx)
            label = label.as_in_context(ctx)

            with autograd.record():
                output, ctx_embedding, q_embedding = net((record_index, q_words, ctx_words, q_chars,
                                                          ctx_chars), ctx_embedding, q_embedding)
                loss = loss_function(output, label)

            loss.backward()
            trainer.step(options.batch_size)

            avg_loss += loss.mean().as_in_context(avg_loss.context)

            # TODO: Update eval metrics calculation with actual predictions
            # eval_metrics.update(label, output)

        # i here would be equal to number of batches
        # if multi-GPU, will also need to multiple by GPU qty
        avg_loss /= i
        epoch_time = time() - e_start
        metrics = eval_metrics.get()
        # TODO: Fix metrics, by using metric.py - original estimator
        # Again, in multi-gpu environment multiple i by GPU qty
        # avg_metrics = [metric / i for metric in metrics[1]]
        # epoch_metrics = (metrics[0], avg_metrics)

        print("\tEPOCH {:2}: train loss {:4.2f} | batch {:4} | lr {:5.3f} | "
              "Time per epoch {:5.2f} seconds"
              .format(i, avg_loss.asscalar(), options.batch_size, trainer.learning_rate,
                      epoch_time))

    print("Training time {:6.2f} seconds".format(time() - train_start))


def get_args():
    parser = argparse.ArgumentParser(description='Question Answering example using BiDAF & SQuAD')
    parser.add_argument('--epochs', type=int, default=40, help='Upper epoch limit')
    parser.add_argument('--embedding_size', type=int, default=100,
                        help='Dimension of the word embedding')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--ctx_embedding_num_layers', type=int, default=2,
                        help='Number of layers in Contextual embedding layer of BiDAF')
    parser.add_argument('--highway_num_layers', type=int, default=2,
                        help='Number of layers in Highway layer of BiDAF')
    parser.add_argument('--modeling_num_layers', type=int, default=2,
                        help='Number of layers in Modeling layer of BiDAF')
    parser.add_argument('--output_num_layers', type=int, default=1,
                        help='Number of layers in Output layer of BiDAF')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--ctx_max_len', type=int, default=400, help='Maximum length of a context')
    # TODO: Question max length in the paper is 30. Had to set it 400 to make dot_product
    # similarity work
    parser.add_argument('--q_max_len', type=int, default=400, help='Maximum length of a question')
    parser.add_argument('--word_max_len', type=int, default=16, help='Maximum characters in a word')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimization algorithm')
    parser.add_argument('--lr', type=float, default=1E-3, help='Initial learning rate')
    parser.add_argument('--lr_update_factor', type=float, default=0.5,
                        help='Learning rate decay factor')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--save_dir', type=str, default='out_dir',
                        help='directory path to save the final model and training log')
    parser.add_argument('--gpu', type=int, default=None,
                        help='id of the gpu to use. Set it to empty means to use cpu.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)
    logging_config(args.save_dir)

    train_dataset, train_dataloader = get_data(is_train=True, options=args)
    word_vocab, char_vocab = get_vocabs(train_dataset, options=args)

    net = BiDAFModel(word_vocab, char_vocab, args, prefix="bidaf")
    net.initialize(init.Xavier(magnitude=2.24))

    run_training(net, train_dataloader, args)
