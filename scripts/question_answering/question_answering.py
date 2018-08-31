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
from mxnet.gluon.nn import HybridSequential

from scripts.question_answering.attention_flow import AttentionFlow
from scripts.question_answering.similarity_function import DotProductSimilarity

__all__ = ['BiDAFEmbedding', 'BiDAFModelingLayer', 'BiDAFOutputLayer', 'BiDAFModel']

from mxnet import nd, init
from mxnet.gluon import Block
from mxnet.gluon import nn
from mxnet.gluon.rnn import LSTM

from gluonnlp.model import ConvolutionalEncoder, Highway


class BiDAFEmbedding(Block):
    """BiDAFEmbedding is a class describing embeddings that are separately applied to question
    and context of the datasource. Both question and context are passed in two NDArrays:
    1. Matrix of words: batch_size x words_per_question/context
    2. Tensor of characters: batch_size x words_per_question/context x chars_per_word
    """
    def __init__(self, word_vocab, char_vocab, contextual_embedding_nlayers=2, highway_nlayers=2,
                 embedding_size=100, prefix=None, params=None):
        super(BiDAFEmbedding, self).__init__(prefix=prefix, params=params)

        self._char_dense_embedding = nn.Embedding(input_dim=len(char_vocab), output_dim=8)
        self._char_conv_embedding = ConvolutionalEncoder(
            embed_size=8,
            num_filters=(100,),
            ngram_filter_sizes=(5,),
            num_highway=None,
            conv_layer_activation='relu',
            output_size=None
        )

        self._word_embedding = nn.Embedding(input_dim=len(word_vocab), output_dim=embedding_size,
                                            weight_initializer=init.Constant(
                                                word_vocab.embedding.idx_to_vec))

        self._highway_network = Highway(2 * embedding_size, num_layers=highway_nlayers)
        self._contextual_embedding = LSTM(hidden_size=embedding_size,
                                          num_layers=contextual_embedding_nlayers,
                                          bidirectional=True)

    def forward(self, x, contextual_embedding_state=None):  # pylint: disable=arguments-differ
        batch_size = x[0].shape[0]
        # Changing shape from NTC to TNC as most MXNet blocks work with TNC format natively
        word_level_data = nd.transpose(x[0], axes=(1, 0))
        char_level_data = nd.transpose(x[1], axes=(1, 0, 2))

        # Get word embeddings. Output is batch_size x seq_len x embedding size (100)
        word_embedded = self._word_embedding(word_level_data)

        # Get char level embedding in multiple steps:
        # Step 1. Embed into 8-dim vector
        char_level_data = self._char_dense_embedding(char_level_data)

        # Step 2. Transpose to put seq_len first axis to later iterate over it
        # In that way we can get embedding per token of every batch
        char_level_data = nd.transpose(char_level_data, axes=(0, 2, 1, 3))

        # Step 3. Iterate over tokens of each batch and apply convolutional encoder
        # As a result of a single iteration, we get token embedding for every batch
        token_list = []
        for token_of_all_batches in char_level_data:
            token_list.append(self._char_conv_embedding(token_of_all_batches))

        # Step 4. Concat all tokens embeddings to create a single tensor.
        char_embedded = nd.concat(*token_list, dim=0)

        # Step 5. Reshape tensor to match dimensions of embedded words
        char_embedded = char_embedded.reshape(shape=word_embedded.shape)

        # Concat embeddings, making channels size = 200
        highway_input = nd.concat(char_embedded, word_embedded, dim=2)
        # Pass through highway, shape remains unchanged
        highway_output = self._highway_network(highway_input)

        # Create starting state if necessary
        contextual_embedding_state = \
            self._contextual_embedding.begin_state(batch_size, ctx=highway_output.context) \
            if contextual_embedding_state is None else contextual_embedding_state

        # Pass through contextual embedding, which is just bi-LSTM
        ce_output, ce_state = self._contextual_embedding(highway_output,
                                                         contextual_embedding_state)

        return ce_output, ce_state


class BiDAFModelingLayer(Block):
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
    def __init__(self, input_dim=100, nlayers=2, biflag=True,
                 dropout=0.2, prefix=None, params=None):
        super(BiDAFModelingLayer, self).__init__(prefix=prefix, params=params)

        self._modeling_layer = LSTM(hidden_size=input_dim, num_layers=nlayers, dropout=dropout,
                                    bidirectional=biflag)

    def forward(self, x):  # pylint: disable=arguments-differ
        out = self._modeling_layer(x)
        return out


class BiDAFOutputLayer(Block):
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
    units : `int`, default 10 * ``span_start_input_dim``
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
    def __init__(self, span_start_input_dim=100, units=None, nlayers=1, biflag=True,
                 dropout=0.2, prefix=None, params=None):
        super(BiDAFOutputLayer, self).__init__(prefix=prefix, params=params)

        units = 10 * span_start_input_dim if units is None else units

        self._start_index_dense = nn.Dense(units=units)
        self._end_index_lstm = LSTM(hidden_size=span_start_input_dim,
                                    num_layers=nlayers, dropout=dropout, bidirectional=biflag)
        self._end_index_dense = nn.Dense(units=units)

    def forward(self, x, m):  # pylint: disable=arguments-differ
        # setting batch size as the first dimension
        start_index_input = nd.transpose(nd.concat(x, m, dim=2), axes=(1, 0, 2))
        start_index_dense_output = self._start_index_dense(start_index_input)

        end_index_input_part = self._end_index_lstm(m)
        end_index_input = nd.transpose(nd.concat(x, end_index_input_part, dim=2),
                                       axes=(1, 0, 2))

        end_index_dense_output = self._end_index_dense(end_index_input)

        # Don't need to apply softmax for training, but do need for prediction
        # Maybe should use autograd properties to check it
        # Will need to reuse it to actually make predictions
        # start_index_softmax_output = start_index_dense_output.softmax(axis=1)
        # start_index = nd.argmax(start_index_softmax_output, axis=1)
        # end_index_softmax_output = end_index_dense_output.softmax(axis=1)
        # end_index = nd.argmax(end_index_softmax_output, axis=1)

        # producing output in shape 2 x batch_size x units
        output = nd.concat(nd.expand_dims(start_index_dense_output, axis=0),
                           nd.expand_dims(end_index_dense_output, axis=0), dim=0)

        # transposing it to batch_size x 2 x units
        return nd.transpose(output, axes=(1, 0, 2))


class BiDAFModel(Block):
    """Bidirectional attention flow model for Question answering
    """

    def __init__(self, word_vocab, char_vocab, options, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)

        with self.name_scope():
            self._ctx_embedding = BiDAFEmbedding(word_vocab, char_vocab,
                                                 options.ctx_embedding_num_layers,
                                                 options.highway_num_layers,
                                                 options.embedding_size,
                                                 prefix="context_embedding")
            self._q_embedding = BiDAFEmbedding(word_vocab, char_vocab,
                                               options.ctx_embedding_num_layers,
                                               options.highway_num_layers,
                                               options.embedding_size,
                                               prefix="question_embedding")
            self._attention_layer = AttentionFlow(DotProductSimilarity())
            self._modeling_layer = BiDAFModelingLayer(input_dim=options.embedding_size,
                                                      nlayers=options.modeling_num_layers,
                                                      dropout=options.dropout)
            self._output_layer = BiDAFOutputLayer(span_start_input_dim=options.embedding_size,
                                                  nlayers=options.output_num_layers,
                                                  dropout=options.dropout)

    def forward(self, x, ctx_embedding_states=None, q_embedding_states=None, *args):
        ctx_embedding_output, ctx_embedding_state = self._ctx_embedding([x[2], x[4]],
                                                                        ctx_embedding_states)
        q_embedding_output, q_embedding_state = self._q_embedding([x[1], x[3]],
                                                                  q_embedding_states)

        attention_layer_output = self._attention_layer(ctx_embedding_output, q_embedding_output)
        modeling_layer_output = self._modeling_layer(attention_layer_output)
        output = self._output_layer(attention_layer_output, modeling_layer_output)

        return output, ctx_embedding_state, q_embedding_state
