r"""
This file contains the train code.
"""
import json
import multiprocessing
import time

import mxnet as mx
from mxnet import autograd, gluon, nd
from mxnet.gluon.data import DataLoader
from tqdm import tqdm

try:
    from data_pipeline import SQuADDataPipeline, SQuADDataLoaderTransformer
except ImportError:
    from .data_pipeline import SQuADDataPipeline, SQuADDataLoaderTransformer

try:
    from config import (TRAIN_PARA_LIMIT, TRAIN_QUES_LIMIT, DEV_PARA_LIMIT, DEV_QUES_LIMIT,
                        ANS_LIMIT, CHAR_COUNT, DIM_WORD_EMBED, GLOVE_FILE_NAME, EPOCHS,
                        SAVE_MODEL_PREFIX_NAME, SAVE_TRAINER_PREFIX_NAME,
                        LAST_GLOBAL_STEP, TRAIN_FLAG, EVALUATE_INTERVAL, BETA2,
                        WORD_EMB_FILE_NAME, NEED_LOAD_TRAINED_MODEL, TRAIN_BATCH_SIZE,
                        ACCUM_AVG_TRAIN_CROSS_ENTROPY, BATCH_TRAIN_CROSS_ENTROPY,
                        CTX, WEIGHT_DECAY, CLIP_GRADIENT, TARGET_TRAINER_FILE_NAME, BETA1,
                        INIT_LEARNING_RATE, EXPONENTIAL_MOVING_AVERAGE_DECAY, EPSILON,
                        TARGET_MODEL_FILE_NAME, WARM_UP_STEPS)
except ImportError:
    from .qanet_config import (TRAIN_PARA_LIMIT, TRAIN_QUES_LIMIT, DEV_PARA_LIMIT, DEV_QUES_LIMIT,
                               ANS_LIMIT, CHAR_LIMIT, DIM_WORD_EMBED, GLOVE_FILE_NAME, EPOCHS,
                               SAVE_MODEL_PREFIX_NAME, SAVE_TRAINER_PREFIX_NAME,
                               LAST_GLOBAL_STEP, TRAIN_FLAG, EVALUATE_INTERVAL, BETA2,
                               WORD_EMB_FILE_NAME, NEED_LOAD_TRAINED_MODEL, TRAIN_BATCH_SIZE,
                               ACCUM_AVG_TRAIN_CROSS_ENTROPY, BATCH_TRAIN_CROSS_ENTROPY,
                               CTX, WEIGHT_DECAY, CLIP_GRADIENT, TARGET_TRAINER_FILE_NAME, BETA1,
                               INIT_LEARNING_RATE, EXPONENTIAL_MOVING_AVERAGE_DECAY, EPSILON,
                               TARGET_MODEL_FILE_NAME, WARM_UP_STEPS)
try:
    from qanet_model import MySoftmaxCrossEntropy, QANet
except ImportError:
    from .qanet_model import MySoftmaxCrossEntropy, QANet

try:
    from qanet_evaluate import evaluate
except ImportError:
    from .qanet_evaluate import evaluate

try:
    from ema import ExponentialMovingAverage
except ImportError:
    from .ema import ExponentialMovingAverage

try:
    from utils import warm_up_lr
except ImportError:
    from .utils import warm_up_lr

mx.random.seed(37)
accum_avg_train_ce = []
batch_train_ce = []
dev_f1 = []
dev_em = []
global_step = LAST_GLOBAL_STEP


def train(model, train_dataloader, dev_dataloader, dev_dataset, dev_json_data, trainer,
          loss_function, ema=None, total_batches=0, padding_token_idx=1):
    r"""
    Train and save.
    """
    for e in tqdm(range(EPOCHS)):
        print('Begin %d/%d epoch...' % (e + 1, EPOCHS))
        train_one_epoch(model, train_dataloader, dev_dataloader, dev_dataset, dev_json_data,
                        trainer, loss_function, ema, total_batches, padding_token_idx)

        # save model after train one epoch
        model.save_parameters(SAVE_MODEL_PREFIX_NAME +
                              time.asctime(time.localtime(time.time())))
        trainer.save_states(SAVE_TRAINER_PREFIX_NAME +
                            time.asctime(time.localtime(time.time())))

        with open(ACCUM_AVG_TRAIN_CROSS_ENTROPY +
                  time.asctime(time.localtime(time.time())), 'w') as f:
            f.write(json.dumps(accum_avg_train_ce))

        with open(BATCH_TRAIN_CROSS_ENTROPY +
                  time.asctime(time.localtime(time.time())), 'w') as f:
            f.write(json.dumps(batch_train_ce))


def train_one_epoch(model, train_dataloader, dev_dataloader, dev_dataset, dev_json_data, trainer,
                    loss_function, ema=None, total_batches=0, padding_token_idx=1):
    r"""
    One train loop.
    """
    total_loss = 0
    step = 0
    global global_step
    for _, context, query, context_char, query_char, begin, end in train_dataloader:
        step += 1
        global_step += 1
        # add evaluate per EVALUATE_INTERVAL batchs
        if global_step % EVALUATE_INTERVAL == 0:
            print('global_step == %d' % global_step)
            print('evaluating dev dataset...')
            f1_score, em_score = evaluate(model, dev_dataloader, dev_dataset, dev_json_data, ema,
                                          padding_token_idx)
            print('dev f1:' + str(f1_score) + 'em:' + str(em_score))
            dev_f1.append([global_step, f1_score])
            dev_em.append([global_step, em_score])

        c_mask = context != padding_token_idx
        q_mask = query != padding_token_idx
        batch_sizes = context.shape[0]

        context = gluon.utils.split_and_load(data=context, ctx_list=CTX)
        c_mask = gluon.utils.split_and_load(data=c_mask, ctx_list=CTX)
        query = gluon.utils.split_and_load(data=query, ctx_list=CTX)
        q_mask = gluon.utils.split_and_load(data=q_mask, ctx_list=CTX)
        context_char = gluon.utils.split_and_load(data=context_char, ctx_list=CTX)
        query_char = gluon.utils.split_and_load(data=query_char, ctx_list=CTX)
        begin = gluon.utils.split_and_load(data=begin, ctx_list=CTX)
        end = gluon.utils.split_and_load(data=end, ctx_list=CTX)

        with autograd.record():
            different_ctx_loss = [loss_function(*model(c, q, cc, qc, cm, qm, b, e))
                                  for c, q, cc, qc, cm, qm, b, e in
                                  zip(context, query, context_char, query_char,
                                      c_mask, q_mask, begin, end)]

            for loss in different_ctx_loss:
                loss.backward()

        if global_step == 1:
            for name, param in model.collect_params().items():
                ema.add(name, param.data(CTX[0]))

        trainer.set_learning_rate(warm_up_lr(INIT_LEARNING_RATE, global_step, WARM_UP_STEPS))
        trainer.allreduce_grads()
        reset_embedding_grad(model)
        tmp = []

        for name, parameter in model.collect_params().items():
            grad = parameter.grad(context[0].context)
            if name == 'qanet0_embedding0_weight':
                grad[0:2] += WEIGHT_DECAY * parameter.data(context[0].context)[0:2]
            else:
                grad += WEIGHT_DECAY * parameter.data(context[0].context)
            tmp.append(grad)

        gluon.utils.clip_global_norm(tmp, CLIP_GRADIENT)
        reset_embedding_grad(model)
        trainer.update(batch_sizes, ignore_stale_grad=True)

        for name, param in model.collect_params().items():
            ema(name, param.data(CTX[0]))

        batch_loss = .0
        for loss in different_ctx_loss:
            batch_loss += loss.mean().asscalar()

        batch_loss /= len(different_ctx_loss)
        total_loss += batch_loss

        batch_train_ce.append([global_step, batch_loss])
        accum_avg_train_ce.append([global_step, total_loss / step])

        print('batch %d/%d, total_loss %.2f, batch_loss %.2f' %
              (step, total_batches, total_loss / step, batch_loss), end='\r', flush=True)
        nd.waitall()


def initial_model_parameters(model, word_embedding):
    r"""
    Initial the word embedding layer.
    """
    model.collect_params().initialize(ctx=CTX)
    model.word_emb[0].weight.set_data(word_embedding)


def reset_embedding_grad(model):
    r"""
    Reset the grad about word embedding layer.
    """
    for ctx in CTX:
        model.word_emb[0].weight.grad(ctx=ctx)[1:] = 0


def main():
    r"""
    Main function.
    """
    pipeline = SQuADDataPipeline(TRAIN_PARA_LIMIT, TRAIN_QUES_LIMIT, DEV_PARA_LIMIT,
                                 DEV_QUES_LIMIT, ANS_LIMIT, CHAR_LIMIT, GLOVE_FILE_NAME)
    train_json, dev_json, train_dataset, dev_dataset, word_vocab, char_vocab = \
        pipeline.get_processed_data(use_spacy=True, shrink_word_vocab=True)

    model = QANet(len(word_vocab), len(char_vocab))
    # initial parameters
    print('Initial parameters...')
    if NEED_LOAD_TRAINED_MODEL:
        model.load_parameters(TARGET_MODEL_FILE_NAME, ctx=CTX)
    else:
        print('Initial model parameters...')
        initial_model_parameters(model, word_vocab.embedding.idx_to_vec)
    print(model)
    if TRAIN_FLAG is True:
        loss_function = MySoftmaxCrossEntropy()

        ema = ExponentialMovingAverage(decay=EXPONENTIAL_MOVING_AVERAGE_DECAY)

        # initial trainer
        trainer = gluon.Trainer(
            model.collect_params(),
            'adam',
            {
                'learning_rate': INIT_LEARNING_RATE,
                'beta1': BETA1,
                'beta2': BETA2,
                'epsilon': EPSILON
            }
        )

        if NEED_LOAD_TRAINED_MODEL:
            trainer.load_states(TARGET_TRAINER_FILE_NAME)

        train_dataloader = DataLoader(train_dataset.transform(SQuADDataLoaderTransformer()),
                                      batch_size=TRAIN_BATCH_SIZE, shuffle=True, last_batch='keep',
                                      num_workers=multiprocessing.cpu_count(), pin_memory=True)
        dev_dataloader = DataLoader(dev_dataset.transform(SQuADDataLoaderTransformer()),
                                    batch_size=TRAIN_BATCH_SIZE, shuffle=False, last_batch='keep',
                                    num_workers=multiprocessing.cpu_count(), pin_memory=True)

        # train
        print('Train...')
        train(model, train_dataloader, dev_dataloader, dev_dataset, dev_json, trainer,
              loss_function, ema, total_batches=len(train_dataset) // TRAIN_BATCH_SIZE,
              padding_token_idx=word_vocab[word_vocab.padding_token])
    else:
        print('Evaluating dev set...')
        dev_dataloader = DataLoader(dev_dataset.transform(SQuADDataLoaderTransformer()),
                                    batch_size=TRAIN_BATCH_SIZE, shuffle=False, last_batch='keep',
                                    num_workers=multiprocessing.cpu_count(), pin_memory=True)

        f1_score, em_score = evaluate(model, dev_dataloader, dev_dataset, dev_json, ema=None,
                                      padding_token_idx=word_vocab[word_vocab.padding_token])
        print('The dev dataset F1 is:%s, and EM is: %s' % (f1_score, em_score))


if __name__ == '__main__':
    main()
