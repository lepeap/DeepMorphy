import os
import pickle
import numpy as np
import tensorflow as tf


def rnn_top(scope_name,
             rnn_rez,
             settings,
             classes_number):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        rez_size = settings['rnn_state_size']
        w_softmax = tf.get_variable("W", (rez_size, classes_number))
        b_softmax = tf.get_variable("b", [classes_number])
        logits = tf.matmul(rnn_rez, w_softmax) + b_softmax

    return logits

def rnn_cell_unit(keep_drop, settings, for_usage):
    cell = tf.contrib.rnn.GRUCell(
        num_units=settings['rnn_state_size']
    )
    if not for_usage:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell,
            input_keep_prob = keep_drop
        )
    return cell

def rnn_cell(keep_drop, settings, for_usage):
    if settings['rnn_layers_count' ] >1:
        cells = []
        for i in range(settings['rnn_layers_count']):
            with tf.variable_scope('RnnUnit_%s' % i, reuse=tf.AUTO_REUSE) as scope:
                cell = rnn_cell_unit(keep_drop, settings, for_usage)
            cells.append(cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    else:
        with tf.variable_scope('RnnUnit', reuse=tf.AUTO_REUSE) as scope:
            cell = rnn_cell_unit(keep_drop, settings, for_usage)

    return cell

def build_rnn(rnn_input, keep_drop, seq_len, settings, initial_state_fw=None, initial_state_bw=None, for_usage=False):
    if settings['rnn_layers_count'] > 1 and initial_state_fw is not None:
        initial_state_fw = tuple([initial_state_fw for i in range(settings['rnn_layers_count'])])

    if settings['rnn_layers_count'] > 1 and initial_state_bw is not None:
        initial_state_bw = tuple([initial_state_bw for i in range(settings['rnn_layers_count'])])

    if settings['rnn_bidirectional']:
        with tf.variable_scope('Rnn', reuse=tf.AUTO_REUSE) as scope:
            with tf.variable_scope('FCell', reuse=tf.AUTO_REUSE) as scope:
                fw_cell = rnn_cell(keep_drop, settings, for_usage)
            with tf.variable_scope('BCell', reuse=tf.AUTO_REUSE) as scope:
                bw_cell = rnn_cell(keep_drop, settings, for_usage)
            _, (final_fw, final_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                   cell_bw=bw_cell,
                                                                   sequence_length=seq_len,
                                                                   initial_state_fw = initial_state_fw,
                                                                   initial_state_bw = initial_state_bw,
                                                                   inputs=rnn_input,
                                                                   dtype=tf.float32)
            if settings['rnn_layers_count'] > 1:
                final_fw = final_fw[-1]
                final_bw = final_bw[-1]

            final_state = tf.add(final_fw, final_bw)
            return final_state

    else:
        with tf.variable_scope('Rnn', reuse=tf.AUTO_REUSE) as scope:
            cell = rnn_cell(keep_drop, settings, for_usage)
            rnn_rez, final_state = tf.nn.dynamic_rnn(cell=cell,
                                                     sequence_length=seq_len,
                                                     initial_state=initial_state_fw,
                                                     inputs=rnn_input,
                                                     dtype=tf.float32)
            return final_state[-1]

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        if any(x for x in grad_and_vars if x[0] is None):
            continue

        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def create_reset_metric(metric, scope='reset_metrics', **metric_args):
    """
    Source: https://github.com/tensorflow/tensorflow/issues/4814#issuecomment-314801758

    Usage:

    epoch_loss, epoch_loss_update, epoch_loss_reset = create_reset_metric(
    tf.contrib.metrics.streaming_mean_squared_error, 'epoch_loss',
    predictions=output, labels=target)

    :param scope:
    :param metric_args:
    :return:
    """
    with tf.variable_scope(scope) as scope:
        metric_op, update_op = metric(**metric_args)
        vars = tf.contrib.framework.get_variables(
            scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(vars)
    return metric_op, update_op, reset_op


def load_dataset(dataset_path, devices_count, type, batch_size, use_weights, gram=None):

    if gram is None:
        gram = "classification"

    path = os.path.join(dataset_path, f"{gram}_{type}_dataset.pkl")
    with open(path, 'rb') as f:
        items = pickle.load(f)

    tttt = 0

    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    cur_step = []
    for batch in batches:
        x = np.stack([item[0][0] for item in batch])
        seq_len = np.asarray([item[0][1] for item in batch], np.int)
        ys = np.asarray([item[1] for item in batch])
        weight = np.asarray([item[2] for item in batch])
        if len(cur_step) == devices_count:
            yield cur_step
            cur_step = []

        ##TODO remove
        #if tttt>12:
        #    return
        #tttt+=1

        cur_step.append(dict(
            x=x,
            seq_len=seq_len,
            ys=ys,
            weight=weight
        ))

    if len(cur_step)==devices_count:
        yield cur_step

