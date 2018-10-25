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

import copy
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


def transform_dataset(dataset, vocab_provider, options, enable_filtering=False):
    """Get transformed dataset

    Parameters
    ----------
    dataset : `Dataset`
        Original dataset
    vocab_provider : `VocabularyProvider`
        Vocabulary provider
    options : `Namespace`
        Data transformation arguments
    enable_filtering : `Bool`
        Remove data that doesn't match BiDAF model requirements

    Returns
    -------
    data : Tuple
        A tuple of dataset, QuestionIdMapper and original json data for evaluation
    """
    tokenizer = vocab_provider.get_tokenizer()
    transformer = SQuADTransform(vocab_provider, options.q_max_len,
                                 options.ctx_max_len, options.word_max_len, options.embedding_size)

    transformed_records = []
    long_context = 0
    long_question = 0

    for i, record in enumerate(dataset):
        if enable_filtering:
            tokenized_question = tokenizer(record[2], lower_case=True)
            # we don't need to dispose of context as long as the answer is still
            # present in the context after it is trimmed
            # tokenized_context = tokenizer(record[3], lower_case=True)
            #
            # if len(tokenized_context) > options.ctx_max_len:
            #     long_context += 1
            #     continue

            # but we don't know if the question is still meaningful
            if len(tokenized_question) > options.q_max_len:
                long_question += 1
                continue

        transformed_record = transformer(*record)

        # if answer end index is after ctx_max_len token or
        # it is after q_max_len token we do not use this record
        if enable_filtering and transformed_record[6][0][1] >= options.ctx_max_len:
            continue

        transformed_records.append(transformed_record)

    processed_dataset = SimpleDataset(transformed_records)
    print("{}/{} records. Too long context {}, too long query {}".format(
        len(processed_dataset), i + 1, long_context, long_question))
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
                            last_batch='rollover',
                            pin_memory=True,
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

    if options.rho:
        hyperparameters["rho"] = options.rho

    trainer = Trainer(net.collect_params(), options.optimizer, hyperparameters,
                      kvstore="device", update_on_kvstore=False)

    if options.resume_training:
        path = os.path.join(options.save_dir,
                            'trainer_epoch{:d}.params'.format(options.resume_training - 1))
        trainer.load_states(path)

    loss_function = SoftmaxCrossEntropyLoss()
    ema = None

    train_start = time()
    avg_loss = mx.nd.zeros((1,), ctx=ctx[0], dtype=options.precision)
    iteration = 1
    max_dev_exact = -1
    max_dev_f1 = -1
    max_iteration = -1
    early_stop_tries = 0

    print("Starting training...")

    for e in range(0 if not options.resume_training else options.resume_training,
                   options.epochs):
        avg_loss *= 0  # Zero average loss of each epoch

        ctx_embedding_begin_state_list = net.ctx_embedding.begin_state(ctx)
        q_embedding_begin_state_list = net.ctx_embedding.begin_state(ctx)
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
                    loss = loss_function(begin, begin_end[0]) + \
                           loss_function(end, begin_end[1])
                    losses.append(loss)

            for loss in losses:
                loss.backward()

            if iteration == 1 and options.use_exponential_moving_average:
                ema = PolyakAveraging(net.collect_params(),
                                      options.exponential_moving_average_weight_decay)

                if options.resume_training:
                    path = os.path.join(options.save_dir, 'ema_epoch{:d}.params'.format(
                        options.resume_training - 1))
                    ema.get_params().load(path)

            # in special mode we collect gradients and apply processing only after
            # predefined number of grad_req_add_mode which acts like batch_size counter
            if options.grad_req_add_mode > 0:
                if not iteration % options.grad_req_add_mode != 0 and \
                       iteration != len(dataloader):
                    iteration += 1
                    continue

            scailing_coeff = len(ctx) * options.batch_size \
                if options.grad_req_add_mode == 0 else options.grad_req_add_mode

            if options.lr_warmup_steps:
                trainer.set_learning_rate(get_learning_rate_per_iteration(iteration, options))

            if options.clip or options.train_unk_token:
                trainer.allreduce_grads()
                gradients = get_gradients(net, ctx[0], options)

                if options.clip:
                    gluon.utils.clip_global_norm(gradients, options.clip)

                if options.train_unk_token:
                    reset_embedding_gradients(net, ctx[0])

                if len(ctx) > 1:
                    # in multi gpu mode we propagate new gradients to the rest of gpus
                    for name, parameter in net.collect_params().items():
                        grads = parameter.list_grad()
                        source = grads[0]
                        destination = grads[1:]

                        for dest in destination:
                            source.copyto(dest)

                trainer.update(scailing_coeff)
            else:
                trainer.step(scailing_coeff)

            if ema is not None:
                ema.update()

            if e == options.epochs - 1 and \
               options.log_interval > 0 and \
               iteration > 0 and iteration % options.log_interval == 0:
                evaluate_options = copy.deepcopy(options)
                evaluate_options.batch_size = 10
                evaluate_options.epochs = iteration
                result = run_evaluate_mode(evaluate_options, net, ema)

                print("Iteration {} evaluation results on dev dataset: {}".format(iteration,
                                                                                  result))
                if options.early_stop:
                    if result["f1"] > max_dev_f1:
                        max_dev_f1 = result["f1"]
                        max_dev_exact = result["exact_match"]
                        max_iteration = iteration
                        early_stop_tries = 0
                    else:
                        if early_stop_tries < options.early_stop:
                            early_stop_tries += 1
                            print("Results decreased for {} times".format(early_stop_tries))
                        else:
                            print("Results decreased for {} times. Stop training. "
                                  "Best results are stored at {} params file. F1={}, EM={}"\
                                  .format(options.early_stop + 1, max_iteration,
                                          max_dev_f1, max_dev_exact))
                            break

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

    if options.resume_training:
        return options.lr

    return min(options.lr, options.lr * (math.log(iteration) / math.log(options.lr_warmup_steps)))


def get_gradients(model, ctx, options):
    """Get gradients and apply gradient decay to all layers if required.

    :param BiDAFModel model: Model in training
    :param ctx: Contexts
    :param NameSpace options: Training options
    :return: Array of gradients
    """
    gradients = []

    for name, parameter in model.collect_params().items():
        if is_fixed_embedding_layer(name) and not options.train_unk_token:
            continue

        grad = parameter.grad(ctx)

        if options.weight_decay:
            if is_fixed_embedding_layer(name):
                grad[0] += options.weight_decay * parameter.data(ctx)[0]
            else:
                grad += options.weight_decay * parameter.data(ctx)

        gradients.append(grad)

    return gradients


def reset_embedding_gradients(model, ctx):
    """Gradients for embedding layer doesn't need to be trained. We train only UNK token of
    embedding if required.

    :param BiDAFModel model: Model in training
    :param ctx: Contexts of training
    """
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


def run_preprocess_mode(options):
    # we use both datasets to create proper vocab
    dataset_train = SQuAD(segment='train')
    dataset_dev = SQuAD(segment='dev')

    vocab_provider = VocabProvider([dataset_train, dataset_dev], options)
    transformed_dataset = transform_dataset(dataset_train, vocab_provider, options=options,
                                            enable_filtering=True)
    save_transformed_dataset(transformed_dataset, options.preprocessed_dataset_path)

    if options.preprocessed_val_dataset_path:
        transformed_dataset = transform_dataset(dataset_dev, vocab_provider, options=options)
        save_transformed_dataset(transformed_dataset, options.preprocessed_val_dataset_path)


def run_training_mode(options):
    dataset = SQuAD(segment='train')
    dataset_val = SQuAD(segment='dev')
    vocab_provider = VocabProvider([dataset, dataset_val], options)

    if options.preprocessed_dataset_path and isfile(options.preprocessed_dataset_path):
        transformed_dataset = load_transformed_dataset(options.preprocessed_dataset_path)
    else:
        transformed_dataset = transform_dataset(dataset, vocab_provider, options=options,
                                                enable_filtering=True)
        save_transformed_dataset(transformed_dataset, options.preprocessed_dataset_path)

    train_dataset, train_dataloader = get_record_per_answer_span(transformed_dataset, options)
    word_vocab, char_vocab = get_vocabs(vocab_provider, options=options)
    ctx = get_context(options)

    net = BiDAFModel(word_vocab, char_vocab, options, prefix="bidaf")
    net.cast(options.precision)
    net.initialize(init.Xavier(), ctx=ctx)
    net.hybridize()

    if options.grad_req_add_mode:
        net.collect_params().setattr('grad_req', 'add')

    if options.resume_training:
        print("Resuming training from {} epoch".format(options.resume_training))
        params_path = os.path.join(options.save_dir,
                                   'epoch{:d}.params'.format(int(options.resume_training) - 1))
        net.load_parameters(params_path, ctx)

    run_training(net, train_dataloader, ctx, options=options)


def run_evaluate_mode(options, existing_net=None, existing_ema=None):
    train_dataset = SQuAD(segment='train')
    dataset = SQuAD(segment='dev')

    vocab_provider = VocabProvider([train_dataset, dataset], options)
    mapper = QuestionIdMapper(dataset)

    if options.preprocessed_val_dataset_path and isfile(options.preprocessed_val_dataset_path):
        transformed_dataset = load_transformed_dataset(options.preprocessed_val_dataset_path)
    else:
        transformed_dataset = transform_dataset(dataset, vocab_provider, options=options)
        save_transformed_dataset(transformed_dataset, options.preprocessed_val_dataset_path)

    word_vocab, char_vocab = get_vocabs(vocab_provider, options=options)
    ctx = get_context(options)

    evaluator = PerformanceEvaluator(BiDAFTokenizer(), transformed_dataset,
                                     dataset._read_data(), mapper)

    net = BiDAFModel(word_vocab, char_vocab, options, prefix="bidaf")

    if options.use_exponential_moving_average:
        if existing_ema is None:
            params_path = os.path.join(options.save_dir,
                                       'ema_epoch{:d}.params'.format(int(options.epochs) - 1))
        else:
            save_ema_parameters(existing_ema, options.epochs, options)
            params_path = os.path.join(options.save_dir,
                                       'ema_epoch{:d}.params'.format(options.epochs))

        net.collect_params().load(params_path, ctx=ctx)
    else:
        if existing_net is None:
            params_path = os.path.join(options.save_dir,
                                       'epoch{:d}.params'.format(int(options.epochs) - 1))
        else:
            save_model_parameters(existing_net, options.epochs, options)
            params_path = os.path.join(options.save_dir,
                                       'epoch{:d}.params'.format(options.epochs))

        net.load_parameters(params_path, ctx=ctx)

    net.hybridize(static_alloc=True)
    return evaluator.evaluate_performance(net, ctx, options)


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
        run_preprocess_mode(args)

    if args.train:
        print("Running in training mode")
        run_training_mode(args)

    if args.evaluate:
        print("Running in evaluation mode")
        result = run_evaluate_mode(args)
        print("Evaluation results on dev dataset: {}".format(result))

