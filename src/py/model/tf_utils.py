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


def rnn_cell_unit(settings, for_usage, keep_drop=None):

    cell = tf.contrib.rnn.GRUCell(
        num_units=settings['rnn_state_size']
    )
    if not for_usage and keep_drop is not None:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell,
            input_keep_prob = keep_drop
        )
    if 'use_residual' in settings and settings['use_residual']:
        cell = tf.nn.rnn_cell.ResidualWrapper(cell)

    return cell


def rnn_cell(settings, for_usage, keep_drop=None):
    if settings['rnn_layers_count'] > 1:
        cells = []
        for i in range(settings['rnn_layers_count']):
            with tf.variable_scope('RnnUnit_%s' % i, reuse=tf.AUTO_REUSE) as scope:
                cell = rnn_cell_unit(settings, for_usage, keep_drop)
            cells.append(cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    else:
        with tf.variable_scope('RnnUnit', reuse=tf.AUTO_REUSE) as scope:
            cell = rnn_cell_unit(settings, for_usage, keep_drop)

    return cell


def build_rnn(rnn_input, keep_drop, seq_len, settings, initial_state_fw=None, initial_state_bw=None, for_usage=False, with_seq=False, top_concat=False):
    if settings['rnn_layers_count'] > 1 and initial_state_fw is not None:
        initial_state_fw = tuple([initial_state_fw for i in range(settings['rnn_layers_count'])])

    if settings['rnn_layers_count'] > 1 and initial_state_bw is not None:
        initial_state_bw = tuple([initial_state_bw for i in range(settings['rnn_layers_count'])])

    if settings['rnn_bidirectional']:
        with tf.variable_scope('Rnn', reuse=tf.AUTO_REUSE) as scope:
            with tf.variable_scope('FCell', reuse=tf.AUTO_REUSE) as scope:
                fw_cell = rnn_cell(settings, for_usage, keep_drop)
            with tf.variable_scope('BCell', reuse=tf.AUTO_REUSE) as scope:
                bw_cell = rnn_cell(settings, for_usage, keep_drop)
            seq_val, (final_fw, final_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                   cell_bw=bw_cell,
                                                                   sequence_length=seq_len,
                                                                   initial_state_fw = initial_state_fw,
                                                                   initial_state_bw = initial_state_bw,
                                                                   inputs=rnn_input,
                                                                   dtype=tf.float32)

            if settings['rnn_layers_count'] > 1:
                final_fw = final_fw[-1]
                final_bw = final_bw[-1]

            if top_concat:
                final_state = tf.concat([final_fw, final_bw], axis=1)
                final_state = tf.layers.dense(final_state, settings['rnn_state_size'])
                seq_val = tf.concat([seq_val[0], seq_val[1]], axis=2)
                seq_val = tf.layers.dense(seq_val, settings['rnn_state_size'])
            else:
                final_state = tf.add(final_fw, final_bw)
                seq_val = tf.add(seq_val[0], seq_val[1])

            if with_seq:
                return final_state, seq_val
            else:
                return final_state

    else:
        with tf.variable_scope('Rnn', reuse=tf.AUTO_REUSE) as scope:
            cell = rnn_cell(settings, for_usage, keep_drop)
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


def create_reset_metric(metric, scope='reset_metrics', *args, **metric_args):
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
        metric_op, update_op = metric(*args, **metric_args)
        vars = tf.contrib.framework.get_variables(
            scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(vars)
    return metric_op, update_op, reset_op


def load_cls_dataset(dataset_path, devices_count, type, batch_size, use_weights, gram="main"):

    path = os.path.join(dataset_path, f"{gram}_{type}_dataset.pkl")
    with open(path, 'rb') as f:
        items = pickle.load(f)

    tttt = 0

    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    cur_step = []
    for batch in batches:
        x = np.stack([item['x'][0] for item in batch])
        seq_len = np.asarray([item['x'][1] for item in batch], np.int)
        y = np.asarray([item['y'] for item in batch])
        weight = np.asarray([item['weight'] for item in batch]) if use_weights else [1 for _ in range(batch)]
        if len(cur_step) == devices_count:
            yield cur_step
            cur_step = []

        ##TODO remove
        #if tttt>12:
        #    return
        #tttt+=1

        cur_step.append(dict(
            x=x,
            x_seq_len=seq_len,
            y=y,
            weight=weight
        ))

    if len(cur_step)==devices_count:
        yield cur_step


def load_lemma_dataset(dataset_path, devices_count, type, batch_size):
    path = os.path.join(dataset_path, f"lemma_{type}_dataset.pkl")
    with open(path, 'rb') as f:
        items = pickle.load(f)

    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    cur_step = []
    for batch in batches:
        x = np.stack([item['x'] for item in batch])
        x_seq_len = np.asarray([item['x_len'] for item in batch], np.int)
        x_cls = np.asarray([item['main_cls'] for item in batch], np.int)
        y_seq_len = np.asarray([item['y_len'] for item in batch], np.int)
        max_len = y_seq_len.max()
        y = np.asarray([item['y'][:max_len] for item in batch])

        x_src = [item['x_src'] for item in batch]
        y_src = [item['y_src'] for item in batch]

        if len(cur_step) == devices_count:
            yield cur_step
            cur_step = []

        cur_step.append(dict(
            x=x,
            x_seq_len=x_seq_len,
            x_cls=x_cls,
            y=y,
            y_seq_len=y_seq_len,
            x_src=x_src,
            y_src=y_src
        ))

    if len(cur_step) == devices_count and all([len(step['x']) == batch_size for step in cur_step]):
        yield cur_step


def load_inflect_dataset(dataset_path, devices_count, type, batch_size):
    path = os.path.join(dataset_path, f"inflect_{type}_dataset.pkl")
    with open(path, 'rb') as f:
        items = pickle.load(f)

    tttt = 0

    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    cur_step = []
    for batch in batches:
        x = np.stack([item['x'] for item in batch])
        x_seq_len = np.asarray([item['x_len'] for item in batch], np.int)
        x_cls = np.asarray([item['x_cls'] for item in batch], np.int)
        y_seq_len = np.asarray([item['y_len'] for item in batch], np.int)
        y_cls = np.asarray([item['y_cls'] for item in batch], np.int)
        max_len = y_seq_len.max()
        y = np.asarray([item['y'][:max_len] for item in batch])
        x_src = [item['x_src'] for item in batch]
        y_src = [item['y_src'] for item in batch]

        if len(cur_step) == devices_count:
            yield cur_step
            cur_step = []

        cur_step.append(dict(
            x=x,
            x_seq_len=x_seq_len,
            x_cls=x_cls,
            y=y,
            y_seq_len=y_seq_len,
            y_cls=y_cls,
            x_src=x_src,
            y_src=y_src
        ))

        #TODO remove
        #if tttt>0:
        #    return
        #tttt+=1

    if len(cur_step) == devices_count and all([len(step['x']) == batch_size for step in cur_step]):
        yield cur_step

def decoder(helper, encoder_outputs, seq_len, settings, chars_count, batch_size, for_usage):
    #attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
    #    num_units=settings['decoder']['rnn_state_size'],
    #    memory=encoder_outputs,
    #    memory_sequence_length=seq_len
    #)

    cell = rnn_cell(settings['decoder'], for_usage)


    #attn_cell = tf.contrib.seq2seq.AttentionWrapper(
    #    cell,
    #    attention_mechanism,
    #    attention_layer_size=settings['decoder']['rnn_state_size']
    #)
    out_cell = tf.contrib.rnn.OutputProjectionWrapper(
        cell,
        chars_count,
    )


    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=out_cell,
        helper=helper,
        initial_state=out_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
    )
    outputs = tf.contrib.seq2seq.dynamic_decode(
        decoder=decoder,
        impute_finished=True,
        maximum_iterations=settings['max_length']
    )
    return outputs[0]


def seq2seq(settings, encoder_input, keep_drop, seq_len, init_state, chars_count, for_usage):
    with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE) as scope:
        encoder_output = build_rnn(encoder_input,
                                   keep_drop,
                                   seq_len,
                                   settings['encoder'],
                                   init_state,
                                   init_state,
                                   for_usage)

    with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE) as scope:

        if for_usage:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(encoder_output, start_token=1, end_token=2)
        else:
            helper = tf.contrib.seq2seq.TrainingHelper(encoder_output, seq_len)

        decoder(helper, encoder_output, seq_len, settings, chars_count, settings['batch_size'], for_usage)


def optimistic_restore(session, save_file):
    # src https://github.com/tensorflow/tensorflow/issues/312#issuecomment-287455836
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)