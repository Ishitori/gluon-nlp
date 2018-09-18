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
import multiprocessing
import os
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
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss

import gluonnlp as nlp
from gluonnlp.data import SQuAD

from scripts.question_answering.data_processing import VocabProvider, SQuADTransform
from scripts.question_answering.performance_evaluator import PerformanceEvaluator
from scripts.question_answering.question_answering import *
from scripts.question_answering.question_id_mapper import QuestionIdMapper
from scripts.question_answering.utils import logging_config

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
                                 options.ctx_max_len, options.word_max_len)
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
        A tuple of dataset and dataloader
    """
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
                            last_batch='keep')

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
        Command arguments

    """
    ctx = []

    if options.gpu is None:
        ctx.append(mx.cpu(0))
        print('Use CPU')
    else:
        indices = options.gpu.split(',')

        for index in indices:
            ctx.append(mx.gpu(int(index)))

    return ctx


def run_training(net, dataloader, evaluator, ctx, options):
    """Main function to do training of the network

    Parameters
    ----------
    net : `Block`
        Network to train
    dataloader : `DataLoader`
        Initialized dataloader
    evaluator: `PerformanceEvaluator`
        Used to plug in official evaluation script
    ctx: `Context`
        Training context
    options : `Namespace`
        Training arguments
    """

    trainer = Trainer(net.collect_params(), args.optimizer,
                      {'learning_rate': options.lr},
                      kvstore="device")
    loss_function = SoftmaxCrossEntropyLoss()

    train_start = time()
    avg_loss = mx.nd.zeros((1,), ctx=ctx[0], dtype=options.precision)
    print("Starting training...")

    for e in range(args.epochs):
        avg_loss *= 0  # Zero average loss of each epoch

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

            for ri, qw, cw, qc, cc, l in zip(record_index, q_words, ctx_words,
                                             q_chars, ctx_chars, label):
                with autograd.record():
                    o, _, _ = net((ri, qw, cw, qc, cc))
                    loss = loss_function(o, l)
                    losses.append(loss)

            for l in losses:
                l.backward()

            trainer.step(options.batch_size)

            for l in losses:
                avg_loss += l.mean().as_in_context(avg_loss.context)

        mx.nd.waitall()
        print("Start evaluate performance")
        #eval_results = evaluator.evaluate_performance(net, ctx, options)
        eval_results = {}
        print("End evaluate performance")

        avg_loss /= (i * len(ctx))

        # block the call here to get correct Time per epoch
        avg_loss_scalar = avg_loss.asscalar()
        epoch_time = time() - e_start

        print("\tEPOCH {:2}: train loss {:4.2f} | batch {:4} | lr {:5.3f} | "
              "Time per epoch {:5.2f} seconds | {}"
              .format(e, avg_loss_scalar, options.batch_size, trainer.learning_rate,
                      epoch_time, eval_results))

        save_model_parameters(net, e, options)

    print("Training time {:6.2f} seconds".format(time() - train_start))


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


def get_args():
    """Get console arguments
    """
    parser = argparse.ArgumentParser(description='Question Answering example using BiDAF & SQuAD')
    parser.add_argument('--preprocess', type=bool, default=False, help='Preprocess dataset only')
    parser.add_argument('--train', type=bool, default=False, help='Run training')
    parser.add_argument('--evaluate', type=bool, default=False, help='Run evaluation on dev dataset')
    parser.add_argument('--preprocessed_dataset_path', type=str,
                        default="preprocessed_dataset.p", help='Path to preprocessed dataset')
    parser.add_argument('--preprocessed_val_dataset_path', type=str,
                        default="preprocessed_val_dataset.p", help='Path to preprocessed '
                                                                   'validation dataset')
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
    parser.add_argument('--q_max_len', type=int, default=30, help='Maximum length of a question')
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
    parser.add_argument('--gpu', type=str, default=None,
                        help='Coma-separated ids of the gpu to use. Empty means to use cpu.')
    parser.add_argument('--precision', type=str, default='float32', choices=['float16', 'float32'],
                        help='Use float16 or float32 precision')
    parser.add_argument('--save_prediction_path', type=str, default='',
                        help='Path to save predictions')
    #parser.add_argument('--use_multiprecision_in_optimizer', type=bool, default=False,
    #                    help='When using float16, shall optimizer use multiprecision.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)
    logging_config(args.save_dir)

    if args.preprocess:
        if not args.preprocessed_dataset_path:
            logging.error("Preprocessed_data_path attribute is not provided")
            exit(1)

        print("Running in preprocessing mode")

        dataset = SQuAD(segment='train')
        vocab_provider = VocabProvider(dataset)
        transformed_dataset = transform_dataset(dataset, vocab_provider, options=args)
        save_transformed_dataset(transformed_dataset, args.preprocessed_dataset_path)
        exit(0)

    if args.train:
        print("Running in training mode")

        dataset = SQuAD(segment='train')
        vocab_provider = VocabProvider(dataset)
        mapper = QuestionIdMapper(dataset)

        if args.preprocessed_dataset_path and isfile(args.preprocessed_dataset_path):
            transformed_dataset = load_transformed_dataset(args.preprocessed_dataset_path)
        else:
            transformed_dataset = transform_dataset(dataset, vocab_provider, options=args)
            save_transformed_dataset(transformed_dataset, args.preprocessed_dataset_path)

        train_dataset, train_dataloader = get_record_per_answer_span(transformed_dataset, args)
        word_vocab, char_vocab = get_vocabs(vocab_provider, options=args)
        ctx = get_context(args)

        evaluator = PerformanceEvaluator(transformed_dataset, dataset._read_data(), mapper)
        net = BiDAFModel(word_vocab, char_vocab, args, prefix="bidaf")
        net.initialize(init.Xavier(magnitude=2.24), ctx=ctx)
        net.cast(args.precision)

        run_training(net, train_dataloader, evaluator, ctx, options=args)

    if args.evaluate:
        print("Running in evaluation mode")
        # we use training dataset to build vocabs
        model_path = os.path.join(args.save_dir, 'epoch{:d}.params'.format(int(args.epochs) - 1))

        train_dataset = SQuAD(segment='train')
        vocab_provider = VocabProvider(train_dataset)

        dataset = SQuAD(segment='dev')
        mapper = QuestionIdMapper(dataset)

        transformed_dataset = load_transformed_dataset(args.preprocessed_val_dataset_path) \
            if args.preprocessed_val_dataset_path and isfile(args.preprocessed_val_dataset_path) \
            else transform_dataset(dataset, vocab_provider, options=args)

        if args.preprocessed_val_dataset_path and isfile(args.preprocessed_val_dataset_path):
            transformed_dataset = load_transformed_dataset(args.preprocessed_val_dataset_path)
        else:
            transformed_dataset = transform_dataset(dataset, vocab_provider, options=args)
            save_transformed_dataset(transformed_dataset, args.preprocessed_val_dataset_path)

        val_dataset, val_dataloader = get_record_per_answer_span(transformed_dataset, args)
        word_vocab, char_vocab = get_vocabs(vocab_provider, options=args)
        ctx = get_context(args)

        evaluator = PerformanceEvaluator(transformed_dataset, dataset._read_data(), mapper)
        net = BiDAFModel(word_vocab, char_vocab, args, prefix="bidaf")
        net.load_parameters(model_path, ctx=ctx)

        result = evaluator.evaluate_performance(net, ctx, args)
        print("Evaluation results on dev dataset: {}".format(result))

