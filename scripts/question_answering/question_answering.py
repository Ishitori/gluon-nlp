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

"""BiDAF model blocks"""
from scripts.question_answering.bidaf import BidirectionalAttentionFlow
from scripts.question_answering.similarity_function import DotProductSimilarity

__all__ = ['BiDAFEmbedding', 'BiDAFModelingLayer', 'BiDAFOutputLayer', 'BiDAFModel']

from mxnet import initializer
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.rnn import LSTM

from gluonnlp.model import ConvolutionalEncoder, Highway


class BiDAFEmbedding(HybridBlock):
    """BiDAFEmbedding is a class describing embeddings that are separately applied to question
    and context of the datasource. Both question and context are passed in two NDArrays:
    1. Matrix of words: batch_size x words_per_question/context
    2. Tensor of characters: batch_size x words_per_question/context x chars_per_word
    """
    def __init__(self, batch_size, word_vocab, char_vocab, max_seq_len,
                 contextual_embedding_nlayers=2, highway_nlayers=2, embedding_size=100,
                 precision='float32', prefix=None, params=None):
        super(BiDAFEmbedding, self).__init__(prefix=prefix, params=params)

        self._word_vocab = word_vocab
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._precision = precision
        self._embedding_size = embedding_size

        with self.name_scope():
            self._char_dense_embedding = nn.Embedding(input_dim=len(char_vocab),
                                                      output_dim=8)
            self._char_conv_embedding = ConvolutionalEncoder(
                embed_size=8,
                num_filters=(100,),
                ngram_filter_sizes=(5,),
                num_highway=None,
                conv_layer_activation='relu',
                output_size=None
            )

            self._word_embedding = nn.Embedding(input_dim=len(word_vocab),
                                                output_dim=embedding_size)

            self._highway_network = Highway(2 * embedding_size, num_layers=highway_nlayers)
            self._contextual_embedding = LSTM(hidden_size=embedding_size,
                                              num_layers=contextual_embedding_nlayers,
                                              bidirectional=True, input_size=2 * embedding_size)

    def initialize(self, init=initializer.Uniform(), ctx=None, verbose=False, force_reinit=False):
        super(BiDAFEmbedding, self).initialize(init, ctx, verbose, force_reinit)
        self._word_embedding.weight.set_data(self._word_vocab.embedding.idx_to_vec)

    def begin_state(self, ctx):
        state_list = [self._contextual_embedding.begin_state(self._batch_size,
                                                             dtype=self._precision,
                                                             ctx=c) for c in ctx]
        return state_list

    def hybrid_forward(self, F, w, c, contextual_embedding_state, *args):
        # Changing shape from NTC to TNC as most MXNet blocks work with TNC format natively
        word_level_data = F.transpose(w, axes=(1, 0))
        char_level_data = F.transpose(c, axes=(1, 0, 2))

        # Get word embeddings. Output is batch_size x seq_len x embedding size (100)
        word_embedded = self._word_embedding(word_level_data)

        # Get char level embedding in multiple steps:
        # Step 1. Embed into 8-dim vector
        char_level_data = self._char_dense_embedding(char_level_data)

        # Step 2. Transpose to put seq_len first axis to later iterate over it
        # In that way we can get embedding per token of every batch
        char_level_data = F.transpose(char_level_data, axes=(0, 2, 1, 3))

        # Step 3. Iterate over tokens of each batch and apply convolutional encoder
        # As a result of a single iteration, we get token embedding for every batch
        def convolute(token_of_all_batches, _):
            return self._char_conv_embedding(token_of_all_batches), []

        char_embedded, _ = F.contrib.foreach(convolute, char_level_data, [])

        # Step 4. Concat all tokens embeddings to create a single tensor.
        # char_embedded = F.concat(*token_list, dim=0)

        # Step 5. Reshape tensor to match dimensions of embedded words
        # char_embedded = char_embedded.reshape(shape=(self._max_seq_len,
        #                                             self._batch_size,
        #                                             self._embedding_size))

        # Concat embeddings, making channels size = 200
        highway_input = F.concat(char_embedded, word_embedded, dim=2)

        # Pass through highway, shape remains unchanged
        highway_output = self._highway_network(highway_input)

        ce_output, ce_state = self._contextual_embedding(highway_output,
                                                         contextual_embedding_state)
        return ce_output


class BiDAFModelingLayer(HybridBlock):
    """BiDAFModelingLayer implements modeling layer of BiDAF paper. It is used to scan over context
    produced by Attentional Flow Layer via 2 layer bi-LSTM.

    The input data for the forward should be of dimension 8 * hidden_size (default hidden_size
    is 100).

    Parameters
    ----------

    input_dim : `int`, default 100
        The number of features in the hidden state h of LSTM
    nlayers : `int`, default 2
        Number of recurrent layers.
    biflag: `bool`, default True
        If `True`, becomes a bidirectional RNN.
    dropout: `float`, default 0
        If non-zero, introduces a dropout layer on the outputs of each
        RNN layer except the last layer.
    prefix : `str` or None
        Prefix of this `Block`.
    params : `ParameterDict` or `None`
        Shared Parameters for this `Block`.
    """
    def __init__(self, batch_size, input_dim=100, nlayers=2, biflag=True,
                 dropout=0.2, precision='float32', prefix=None, params=None):
        super(BiDAFModelingLayer, self).__init__(prefix=prefix, params=params)

        self._batch_size = batch_size
        self._precision = precision

        with self.name_scope():
            self._modeling_layer = LSTM(hidden_size=input_dim, num_layers=nlayers, dropout=dropout,
                                        bidirectional=biflag, input_size=800)

    def begin_state(self, ctx):
        state_list = [self._modeling_layer.begin_state(self._batch_size,
                                                       dtype=self._precision,
                                                       ctx=c) for c in ctx]
        return state_list

    def hybrid_forward(self, F, x, state, *args):
        out, _ = self._modeling_layer(x, state)
        return out


class BiDAFOutputLayer(HybridBlock):
    """
    ``BiDAFOutputLayer`` produces the final prediction of an answer. The output is a tuple of
    start index and end index of the answer in the paragraph per each batch.

    It accepts 2 inputs:
        `x` : the output of Attention layer of shape:
        seq_max_length x batch_size x 8 * span_start_input_dim

        `m` : the output of Modeling layer of shape:
         seq_max_length x batch_size x 2 * span_start_input_dim

    Parameters
    ----------
    span_start_input_dim : `int`, default 100
        The number of features in the hidden state h of LSTM
    units : `int`, default 4 * ``span_start_input_dim``
        Number of hidden units of `Dense` layer
    nlayers : `int`, default 1
        Number of recurrent layers.
    biflag: `bool`, default True
        If `True`, becomes a bidirectional RNN.
    dropout: `float`, default 0
        If non-zero, introduces a dropout layer on the outputs of each
        RNN layer except the last layer.
    prefix : `str` or None
        Prefix of this `Block`.
    params : `ParameterDict` or `None`
        Shared Parameters for this `Block`.
    """
    def __init__(self, batch_size, span_start_input_dim=100, units=None, nlayers=1, biflag=True,
                 dropout=0.2, precision='float32', prefix=None, params=None):
        super(BiDAFOutputLayer, self).__init__(prefix=prefix, params=params)

        units = 4 * span_start_input_dim if units is None else units

        self._batch_size = batch_size
        self._precision = precision

        with self.name_scope():
            self._start_index_dense = nn.Dense(units=units, in_units=400000)
            self._end_index_lstm = LSTM(hidden_size=span_start_input_dim,
                                        num_layers=nlayers, dropout=dropout, bidirectional=biflag,
                                        input_size=200)
            self._end_index_dense = nn.Dense(units=units, in_units=400000)

    def begin_state(self, ctx):
        state_list = [self._end_index_lstm.begin_state(self._batch_size,
                                                       dtype=self._precision,
                                                       ctx=c) for c in ctx]
        return state_list

    def hybrid_forward(self, F, x, m, state, *args):  # pylint: disable=arguments-differ

        # setting batch size as the first dimension
        start_index_input = F.transpose(F.concat(x, m, dim=2), axes=(1, 0, 2))
        start_index_dense_output = self._start_index_dense(start_index_input)

        end_index_input_part, _ = self._end_index_lstm(m, state)
        end_index_input = F.transpose(F.concat(x, end_index_input_part, dim=2),
                                       axes=(1, 0, 2))

        end_index_dense_output = self._end_index_dense(end_index_input)

        # Don't need to apply softmax for training, but do need for prediction
        # Maybe should use autograd properties to check it
        # Will need to reuse it to actually make predictions
        # start_index_softmax_output = start_index_dense_output.softmax(axis=1)
        # start_index = F.argmax(start_index_softmax_output, axis=1)
        # end_index_softmax_output = end_index_dense_output.softmax(axis=1)
        # end_index = F.argmax(end_index_softmax_output, axis=1)

        # producing output in shape 2 x batch_size x units
        output = F.concat(F.expand_dims(start_index_dense_output, axis=0),
                           F.expand_dims(end_index_dense_output, axis=0), dim=0)

        # transposing it to batch_size x 2 x units
        return F.transpose(output, axes=(1, 0, 2))


class BiDAFModel(HybridBlock):
    """Bidirectional attention flow model for Question answering
    """
    def __init__(self, word_vocab, char_vocab, options, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self._options = options

        with self.name_scope():
            self.ctx_embedding = BiDAFEmbedding(options.batch_size,
                                                word_vocab,
                                                char_vocab,
                                                options.ctx_max_len,
                                                options.ctx_embedding_num_layers,
                                                options.highway_num_layers,
                                                options.embedding_size,
                                                precision=options.precision,
                                                prefix="context_embedding")
            self.q_embedding = BiDAFEmbedding(options.batch_size,
                                              word_vocab,
                                              char_vocab,
                                              options.q_max_len,
                                              options.ctx_embedding_num_layers,
                                              options.highway_num_layers,
                                              options.embedding_size,
                                              precision=options.precision,
                                              prefix="question_embedding")

            # we multiple embedding_size by 2 because we use bidirectional embedding
            self.attention_layer = BidirectionalAttentionFlow(DotProductSimilarity(),
                                                              options.batch_size,
                                                              options.ctx_max_len,
                                                              options.q_max_len,
                                                              2 * options.embedding_size)
            self.modeling_layer = BiDAFModelingLayer(options.batch_size,
                                                     input_dim=options.embedding_size,
                                                     nlayers=options.modeling_num_layers,
                                                     dropout=options.dropout,
                                                     precision=options.precision)
            self.output_layer = BiDAFOutputLayer(options.batch_size,
                                                 span_start_input_dim=options.embedding_size,
                                                 nlayers=options.output_num_layers,
                                                 dropout=options.dropout,
                                                 precision=options.precision)

    def hybrid_forward(self, F, qw, cw, qc, cc,
                       ctx_embedding_states=None,
                       q_embedding_states=None,
                       modeling_layer_states=None,
                       output_layer_states=None,
                       *args):
        ctx_embedding_output = self.ctx_embedding(cw, cc, ctx_embedding_states)
        q_embedding_output = self.q_embedding(qw, qc, q_embedding_states)

        # attention layer expect batch_size x seq_length x channels
        ctx_embedding_output = F.transpose(ctx_embedding_output, axes=(1, 0, 2))
        q_embedding_output = F.transpose(q_embedding_output, axes=(1, 0, 2))

        # Both masks can be None
        q_mask = qw != 0
        ctx_mask = cw != 0

        attention_layer_output = self.attention_layer(ctx_embedding_output,
                                                      q_embedding_output,
                                                      q_mask,
                                                      ctx_mask)
        attention_layer_output = F.transpose(attention_layer_output, axes=(1, 0, 2))

        # modeling layer expects seq_length x batch_size x channels
        modeling_layer_output = self.modeling_layer(attention_layer_output, modeling_layer_states)

        output = self.output_layer(attention_layer_output, modeling_layer_output,
                                   output_layer_states)

        return output