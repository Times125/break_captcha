#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: _defined
@Time:  2019/8/19 18:49
@Description: overwrite Keras's ctc cost and ctc decode
"""

from tensorflow.python.keras import (backend_config, backend)
from settings import config

__all__ = ['ctc_batch_cost', 'ctc_decode']


def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    """Runs CTC loss algorithm on each batch element.

    Arguments:
        y_true: tensor `(samples, max_string_length)`
            containing the truth labels.
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, 1)` containing the sequence length for
            each batch item in `y_pred`.
        label_length: tensor `(samples, 1)` containing the sequence length for
            each batch item in `y_true`.

    Returns:
        Tensor with shape (samples,1) containing the
            CTC loss of each element.
    """
    label_length = backend.math_ops.cast(
        backend.array_ops.squeeze(label_length, axis=-1), backend.dtypes_module.int32)
    input_length = backend.math_ops.cast(
        backend.array_ops.squeeze(input_length, axis=-1), backend.dtypes_module.int32)
    sparse_labels = backend.math_ops.cast(
        backend.ctc_label_dense_to_sparse(y_true, label_length), backend.dtypes_module.int32)

    y_pred = backend.math_ops.log(backend.array_ops.transpose(y_pred, perm=[1, 0, 2]) + backend_config.epsilon())

    # overwrite here
    return backend.array_ops.expand_dims(
        backend.ctc.ctc_loss(
            inputs=y_pred, labels=sparse_labels, sequence_length=input_length,
            preprocess_collapse_repeated=config.preprocess_collapse_repeated,
            ctc_merge_repeated=config.ctc_merge_repeated,
            time_major=config.time_major), 1)


def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1, merge_repeated=False):
    """Decodes the output of a softmax.

    Can use either greedy search (also known as best path)
    or a constrained dictionary search.

    Arguments:
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, )` containing the sequence length for
            each batch item in `y_pred`.
        greedy: perform much faster best-path search if `true`.
            This does not use a dictionary.
        beam_width: if `greedy` is `false`: a beam search decoder will be used
            with a beam of this width.
        top_paths: if `greedy` is `false`,
            how many of the most probable paths will be returned.
        merge_repeated: If `merge_repeated` is `True`,
            merge repeated classes in output, default 'false'

    Returns:
        Tuple:
            List: if `greedy` is `true`, returns a list of one element that
                contains the decoded sequence.
                If `false`, returns the `top_paths` most probable
                decoded sequences.
                Important: blank labels are returned as `-1`.
            Tensor `(top_paths, )` that contains
                the log probability of each decoded sequence.
    """
    y_pred = backend.math_ops.log(backend.array_ops.transpose(y_pred, perm=[1, 0, 2]) + backend.epsilon())
    input_length = backend.math_ops.cast(input_length, backend.dtypes_module.int32)

    if greedy:
        (decoded, log_prob) = backend.ctc.ctc_greedy_decoder(
            inputs=y_pred, sequence_length=input_length, merge_repeated=merge_repeated)
    else:
        (decoded, log_prob) = backend.ctc.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            beam_width=beam_width,
            top_paths=top_paths,
            merge_repeated=merge_repeated)
    decoded_dense = [
        backend.sparse_ops.sparse_to_dense(
            st.indices, st.dense_shape, st.values, default_value=-1)
        for st in decoded
    ]
    return decoded_dense, log_prob
