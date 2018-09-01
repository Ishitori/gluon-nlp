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

"""Performance evaluator - a proxy class used for plugging in official validation script"""
from mxnet import nd, gluon
from mxnet.gluon.data import DataLoader, ArrayDataset
from scripts.question_answering.metric import evaluate


class PerformanceEvaluator:
    def __init__(self, evaluation_dataset, json_data, question_id_mapper):
        self._evaluation_dataset = evaluation_dataset
        self._json_data = json_data
        self._mapper = question_id_mapper

    def evaluate_performance(self, net, ctx, options):
        """Get results of evaluation by official evaluation script

        Parameters
        ----------
        net : `Block`
            Network
        ctx : `Context`
            Execution context
        options : `Namespace`
            Training arguments

        Returns
        -------
        data : `dict`
            Returns a dictionary of {'exact_match': <value>, 'f1': <value>}
        """

        pred = {}
        eval_dataset = ArrayDataset([(self._mapper.question_id_to_idx[r[1]], r[2], r[3], r[4], r[5])
                        for r in self._evaluation_dataset])
        eval_dataloader = DataLoader(eval_dataset, batch_size=options.batch_size, last_batch='keep')

        for i, data in enumerate(eval_dataloader):
            record_index, q_words, ctx_words, q_chars, ctx_chars = data
            record_index = gluon.utils.split_and_load(record_index, ctx, even_split=False)
            q_words = gluon.utils.split_and_load(q_words, ctx, even_split=False)
            ctx_words = gluon.utils.split_and_load(ctx_words, ctx, even_split=False)
            q_chars = gluon.utils.split_and_load(q_chars, ctx, even_split=False)
            ctx_chars = gluon.utils.split_and_load(ctx_chars, ctx, even_split=False)

            for ri, qw, cw, qc, cc in zip(record_index, q_words, ctx_words, q_chars, ctx_chars):
                out, _, _ = net((ri, qw, cw, qc, cc))
                out_per_index = out.transpose(axes=(1, 0, 2))
                start_indices = PerformanceEvaluator._get_index(out_per_index[0])
                end_indices = PerformanceEvaluator._get_index(out_per_index[1])

                # iterate over batches
                for idx, start, end in zip(data[0], start_indices, end_indices):
                    idx = int(idx.asscalar())
                    start = int(start.asscalar())
                    end = int(end.asscalar())
                    pred[self._mapper.idx_to_question_id[idx]] = self.get_text_result(idx,
                                                                                      (start, end))

        return evaluate(self._json_data['data'], pred)

    def get_text_result(self, idx, answer_span):
        """Converts answer span into actual text from paragraph

        Parameters
        ----------
        idx : `int`
            Question index
        answer_span : `Tuple`
            Answer span (start_index, end_index)

        Returns
        -------
        text : `str`
            A chunk of text for provided answer_span or None if answer span cannot be provided
        """

        start, end = answer_span

        if start > end:
            return None

        question_id = self._mapper.idx_to_question_id[idx]
        context = self._mapper.question_id_to_context[question_id]

        # start index is above the context length - return cannot provide an answer
        if start > len(context) - 1:
            return ''

        # end index is above the context length - let's take answer to the end of the context
        if end > len(context) - 1:
            end = len(context) - 1

        text = ' '.join(context.split()[start:end + 1])
        return text

    @staticmethod
    def _get_index(prediction):
        """Convert prediction to actual index in text

        Parameters
        ----------
        prediction : `NDArray`
            Output of the network

        Returns
        -------
        indices : `NDArray`
            Indices of a word in context for whole batch
        """
        indices_softmax_output = prediction.softmax(axis=1)
        indices = nd.argmax(indices_softmax_output, axis=1)
        return indices
