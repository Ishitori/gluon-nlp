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
import math

import multiprocessing
import os
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from os.path import isfile

import logging
import pickle

import argparse
import numpy as np
import random
from time import time

import mxnet as mx
from mxnet import gluon, init, autograd
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader, SimpleDataset, ArrayDataset

from gluonnlp.data import SQuAD

from scripts.question_answering.data_processing import VocabProvider, SQuADTransform
from scripts.question_answering.exponential_moving_average import PolyakAveraging
from scripts.question_answering.performance_evaluator import PerformanceEvaluator
from scripts.question_answering.question_answering import *
from scripts.question_answering.question_id_mapper import QuestionIdMapper
from scripts.question_answering.tokenizer import BiDAFTokenizer
from scripts.question_answering.utils import logging_config, get_args

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)


def transform_dataset(dataset, vocab_provider, options):
    """Get transformed dataset

    Parameters
    ----------
    dataset : `Dataset`
        Original dataset
    vocab_provider : `VocabularyProvider`
        Vocabulary provider
    options : `Namespace`
        Data transformation arguments

    Returns
    -------
    data : Tuple
        A tuple of dataset, QuestionIdMapper and original json data for evaluation
    """
    transformer = SQuADTransform(vocab_provider, options.q_max_len,
                                 options.ctx_max_len, options.word_max_len, args.embedding_size)
    processed_dataset = SimpleDataset([transformer(*record) for i, record in enumerate(dataset)])
    return processed_dataset


def get_record_per_answer_span(processed_dataset, options):
    """Each record has multiple answers and for training purposes it is better to increase number of
    records by creating a record per each answer.

    Parameters
    ----------
    processed_dataset : `Dataset`
        Transformed dataset, ready to be trained on
    options : `Namespace`
        Command arguments

    Returns
    -------
    data : Tuple
        A tuple of dataset and dataloader. Each item in dataset is:
        (index, question_word_index, context_word_index, question_char_index, context_char_index,
        answers)
    """
    data_no_label = []
    labels = []
    global_index = 0

    # copy records to a record per answer
    for r in processed_dataset:
        # creating a set out of answer_span will deduplicate them
        for answer_span in set(r[6]):
            # if after all preprocessing the answer is not in the context anymore,
            # the item is filtered out
            if options.filter_long_context and (answer_span[0] > r[3].size or
                                                answer_span[1] > r[3].size):
                continue
            # need to remove question id before feeding the data to data loader
            # And I also replace index with global_index when unrolling answers
            data_no_label.append((global_index, r[2], r[3], r[4], r[5]))
            labels.append(mx.nd.array(answer_span))
            global_index += 1

    loadable_data = ArrayDataset(data_no_label, labels)
    dataloader = DataLoader(loadable_data,
                            batch_size=options.batch_size * len(get_context(options)),
                            shuffle=True,
                            last_batch='discard',
                            num_workers=(multiprocessing.cpu_count() -
                                         len(get_context(options)) - 2))

    print("Total records for training: {}".format(len(labels)))
    return loadable_data, dataloader


def get_vocabs(vocab_provider, options):
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
    word_vocab = vocab_provider.get_word_level_vocab(options.embedding_size)
    char_vocab = vocab_provider.get_char_level_vocab()
    return word_vocab, char_vocab


def get_context(options):
    """Return context list to work on

    Parameters
    ----------
    options : `Namespace`
        Command arguments

    """
    ctx = []

    if options.gpu is None:
        ctx.append(mx.cpu(0))
        ctx.append(mx.cpu(1))
        print('Use CPU')
    else:
        indices = options.gpu.split(',')

        for index in indices:
            ctx.append(mx.gpu(int(index)))

    return ctx


def run_training(net, dataloader, ctx, options):
    """Main function to do training of the network

    Parameters
    ----------
    net : `Block`
        Network to train
    dataloader : `DataLoader`
        Initialized dataloader
    ctx: `Context`
        Training context
    options : `Namespace`
        Training arguments
    """

    hyperparameters = {'learning_rate': options.lr}

    if options.precision == 'float16' and options.use_multiprecision_in_optimizer:
        hyperparameters["multi_precision"] = True

    trainer = Trainer(net.collect_params(), args.optimizer, hyperparameters, kvstore="local")
    loss_function = SoftmaxCrossEntropyLoss()
    ema = None

    train_start = time()
    avg_loss = mx.nd.zeros((1,), ctx=ctx[0], dtype=options.precision)
    iteration = 1
    print("Starting training...")

    for e in range(args.epochs):
        avg_loss *= 0  # Zero average loss of each epoch

        ctx_embedding_begin_state_list = net.ctx_embedding.begin_state(ctx)
        q_embedding_begin_state_list = net.q_embedding.begin_state(ctx)
        m_layer_begin_state_list = net.modeling_layer.begin_state(ctx)
        o_layer_begin_state_list = net.output_layer.begin_state(ctx)

        for i, (data, label) in enumerate(dataloader):
            # start timing for the first batch of epoch
            if i == 0:
                e_start = time()

            record_index, q_words, ctx_words, q_chars, ctx_chars = data

            record_index = record_index.astype(options.precision)
            q_words = q_words.astype(options.precision)
            ctx_words = ctx_words.astype(options.precision)
            q_chars = q_chars.astype(options.precision)
            ctx_chars = ctx_chars.astype(options.precision)
            label = label.astype(options.precision)

            record_index = gluon.utils.split_and_load(record_index, ctx, even_split=False)
            q_words = gluon.utils.split_and_load(q_words, ctx, even_split=False)
            ctx_words = gluon.utils.split_and_load(ctx_words, ctx, even_split=False)
            q_chars = gluon.utils.split_and_load(q_chars, ctx, even_split=False)
            ctx_chars = gluon.utils.split_and_load(ctx_chars, ctx, even_split=False)
            label = gluon.utils.split_and_load(label, ctx, even_split=False)

            # Wait for completion of previous iteration to avoid unnecessary memory allocation
            mx.nd.waitall()
            losses = []

            for ri, qw, cw, qc, cc, l, ctx_embedding_begin_state, \
                q_embedding_begin_state, m_layer_begin_state, \
                o_layer_begin_state in zip(record_index, q_words, ctx_words,
                                           q_chars, ctx_chars, label,
                                           ctx_embedding_begin_state_list,
                                           q_embedding_begin_state_list,
                                           m_layer_begin_state_list,
                                           o_layer_begin_state_list):
                with autograd.record():
                    begin, end = net(qw, cw, qc, cc,
                                     ctx_embedding_begin_state,
                                     q_embedding_begin_state,
                                     m_layer_begin_state,
                                     o_layer_begin_state)
                    begin_end = l.split(axis=1, num_outputs=2, squeeze_axis=1)
                    loss = loss_function(begin, begin_end[0]) + loss_function(end, begin_end[1])
                    losses.append(loss)

            for loss in losses:
                loss.backward()

            if iteration == 1 and args.use_exponential_moving_average:
                ema = PolyakAveraging(net.collect_params(),
                                      args.exponential_moving_average_weight_decay)

            trainer.set_learning_rate(get_learning_rate_per_iteration(iteration, options))
            trainer.allreduce_grads()

            gradients = decay_gradients(net, ctx[0], options)
            gluon.utils.clip_global_norm(gradients, options.clip, check_isfinite=False)
            reset_embedding_gradients(net, ctx[0])

            for name, parameter in net.collect_params().items():
                grads = parameter.list_grad()
                source = grads[0]
                destination = grads[1:]

                for dest in destination:
                    source.copyto(dest)

            trainer.update(len(ctx) * options.batch_size, ignore_stale_grad=True)

            if ema is not None:
                ema.update()

            for l in losses:
                avg_loss += l.mean().as_in_context(avg_loss.context)

            iteration += 1

        mx.nd.waitall()

        avg_loss /= (i * len(ctx))

        # block the call here to get correct Time per epoch
        avg_loss_scalar = avg_loss.asscalar()
        epoch_time = time() - e_start

        print("\tEPOCH {:2}: train loss {:6.4f} | batch {:4} | lr {:5.3f} | "
              "Time per epoch {:5.2f} seconds"
              .format(e, avg_loss_scalar, options.batch_size, trainer.learning_rate,
                      epoch_time))

        save_model_parameters(net, e, options)
        save_ema_parameters(ema, e, options)
        save_trainer_parameters(trainer, e, options)

    print("Training time {:6.2f} seconds".format(time() - train_start))


def get_learning_rate_per_iteration(iteration, options):
    """Returns learning rate based on current iteration. Used to implement learning rate warm up
    technique

    :param int iteration: Number of iteration
    :param NameSpace options: Training options
    :return float: learning rate
    """
    return min(options.lr, options.lr * (math.log(iteration) / math.log(options.lr_warmup_steps)))


def decay_gradients(model, ctx, options):
    """Apply gradient decay to all layers. For predefined embedding layers, we train only
    OOV token embeddings

    :param BiDAFModel model: Model in training
    :param ctx: Contexts
    :param NameSpace options: Training options
    :return: Array of gradients
    """
    gradients = []

    for name, parameter in model.collect_params().items():
        grad = parameter.grad(ctx)

        # we train OOV token
        if is_fixed_embedding_layer(name):
            grad[0] += options.weight_decay * parameter.data(ctx)[0]
        else:
            grad += options.weight_decay * parameter.data(ctx)
        gradients.append(grad)

    return gradients


def reset_embedding_gradients(model, ctx):
    """Gradients for glove layers of both question and context embeddings doesn't need to be
    trainer. We train only OOV token embedding.

    :param BiDAFModel model: Model in training
    :param ctx: Contexts of training
    """
    model.q_embedding._word_embedding.weight.grad(ctx=ctx)[1:] = 0
    model.ctx_embedding._word_embedding.weight.grad(ctx=ctx)[1:] = 0


def is_fixed_embedding_layer(name):
    return True if "predefined_embedding_layer" in name else False


def save_model_parameters(net, epoch, options):
    """Save parameters of the trained model

    Parameters
    ----------
    net : `Block`
        Model with trained parameters
    epoch : `int`
        Number of epoch
    options : `Namespace`
        Saving arguments
    """

    if not os.path.exists(options.save_dir):
        os.mkdir(options.save_dir)

    save_path = os.path.join(options.save_dir, 'epoch{:d}.params'.format(epoch))
    net.save_parameters(save_path)


def save_ema_parameters(ema, epoch, options):
    """Save exponentially averaged parameters of the trained model

    Parameters
    ----------
    ema : `PolyakAveraging`
        Model with trained parameters
    epoch : `int`
        Number of epoch
    options : `Namespace`
        Saving arguments
    """
    if ema is None:
        return

    if not os.path.exists(options.save_dir):
        os.mkdir(options.save_dir)

    save_path = os.path.join(options.save_dir, 'ema_epoch{:d}.params'.format(epoch))
    ema.get_params().save(save_path)


def save_trainer_parameters(trainer, epoch, options):
    """Save exponentially averaged parameters of the trained model

    Parameters
    ----------
    trainer : `Trainer`
        Trainer
    epoch : `int`
        Number of epoch
    options : `Namespace`
        Saving arguments
    """
    if trainer is None:
        return

    if not os.path.exists(options.save_dir):
        os.mkdir(options.save_dir)

    save_path = os.path.join(options.save_dir, 'trainer_epoch{:d}.params'.format(epoch))
    trainer.save_states(save_path)


def save_transformed_dataset(dataset, path):
    """Save processed dataset into a file.

    Parameters
    ----------
    dataset : `Dataset`
        Dataset to save
    path : `str`
        Saving path
    """
    pickle.dump(dataset, open(path, "wb"))


def load_transformed_dataset(path):
    """Loads already preprocessed dataset from disk

    Parameters
    ----------
    path : `str`
        Loading path
    """
    processed_dataset = pickle.load(open(path, "rb"))
    return processed_dataset


if __name__ == "__main__":
    args = get_args()
    args.batch_size = int(args.batch_size / len(get_context(args)))
    print(args)
    logging_config(args.save_dir)

    if args.preprocess:
        if not args.preprocessed_dataset_path:
            logging.error("Preprocessed_data_path attribute is not provided")
            exit(1)

        print("Running in preprocessing mode")

        # we use both datasets to create proper vocab
        dataset_train = SQuAD(segment='train')
        dataset_dev = SQuAD(segment='dev')

        vocab_provider = VocabProvider([dataset_train, dataset_dev], args)
        transformed_dataset = transform_dataset(dataset_train, vocab_provider, options=args)
        save_transformed_dataset(transformed_dataset, args.preprocessed_dataset_path)

        if args.preprocessed_val_dataset_path:
            transformed_dataset = transform_dataset(dataset_dev, vocab_provider, options=args)
            save_transformed_dataset(transformed_dataset, args.preprocessed_val_dataset_path)

        exit(0)

    if args.train:
        print("Running in training mode")

        dataset = SQuAD(segment='train')
        dataset_val = SQuAD(segment='dev')
        vocab_provider = VocabProvider([dataset, dataset_val], args)

        if args.preprocessed_dataset_path and isfile(args.preprocessed_dataset_path):
            transformed_dataset = load_transformed_dataset(args.preprocessed_dataset_path)
        else:
            transformed_dataset = transform_dataset(dataset, vocab_provider, options=args)
            save_transformed_dataset(transformed_dataset, args.preprocessed_dataset_path)

        train_dataset, train_dataloader = get_record_per_answer_span(transformed_dataset, args)
        word_vocab, char_vocab = get_vocabs(vocab_provider, options=args)
        ctx = get_context(args)

        net = BiDAFModel(word_vocab, char_vocab, args, prefix="bidaf")
        net.cast(args.precision)
        net.initialize(init.Xavier(magnitude=2.24), ctx=ctx)
        net.hybridize()

        run_training(net, train_dataloader, ctx, options=args)

    if args.evaluate:
        print("Running in evaluation mode")

        train_dataset = SQuAD(segment='train')
        dataset = SQuAD(segment='dev')

        vocab_provider = VocabProvider([train_dataset, dataset], args)
        mapper = QuestionIdMapper(dataset)

        if args.preprocessed_val_dataset_path and isfile(args.preprocessed_val_dataset_path):
            transformed_dataset = load_transformed_dataset(args.preprocessed_val_dataset_path)
        else:
            transformed_dataset = transform_dataset(dataset, vocab_provider, options=args)
            save_transformed_dataset(transformed_dataset, args.preprocessed_val_dataset_path)

        word_vocab, char_vocab = get_vocabs(vocab_provider, options=args)
        ctx = get_context(args)

        evaluator = PerformanceEvaluator(BiDAFTokenizer(), transformed_dataset,
                                         dataset._read_data(), mapper)
        net = BiDAFModel(word_vocab, char_vocab, args, prefix="bidaf")

        if args.use_exponential_moving_average:
            params_path = os.path.join(args.save_dir,
                                      'ema_epoch{:d}.params'.format(int(args.epochs) - 1))
            net.collect_params().load(params_path, ctx=ctx)
        else:
            params_path = os.path.join(args.save_dir,
                                      'epoch{:d}.params'.format(int(args.epochs) - 1))
            net.load_parameters(params_path, ctx=ctx)

        net.hybridize(static_alloc=True)

        result = evaluator.evaluate_performance(net, ctx, args)
        print("Evaluation results on dev dataset: {}".format(result))
