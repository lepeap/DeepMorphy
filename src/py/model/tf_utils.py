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

    if len(cur_step) == devices_count and all([len(step['x']) == batch_size for step in cur_step]):
        yield cur_step


def seq2seq(graph_part,
            batch_size,
            x,
            x_init,
            x_seq_len,
            y,
            y_init,
            y_seq_len):
    start_tokens = tf.fill([batch_size], graph_part.start_char_index)
    initializer = tf.contrib.layers.xavier_initializer()
    y_seq_len += 1

    with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
        encoder_char_embeddings = tf.get_variable(
            "CharEmbeddings",
            [graph_part.chars_count, graph_part.settings['char_vector_size']],
            initializer=initializer
        )

        encoder_init_state = ClsGramEmbedder(graph_part.main_cls_dic,
                                          graph_part.settings['encoder']['gram_vector_size'],
                                          graph_part.settings['encoder']['ad_cls_vector_size'])(x_init, batch_size)

        #encoder_cls_emb = tf.get_variable(
        #    "XInitEmbeddings",
        #    (graph_part.main_classes_count, graph_part.settings['encoder']['rnn_state_size']),
        #    initializer=initializer
        #)
        #encoder_init_state = tf.nn.embedding_lookup(encoder_cls_emb, x_init)

        encoder_input = tf.nn.embedding_lookup(encoder_char_embeddings, x)

    with tf.variable_scope("Decoder", reuse=tf.AUTO_REUSE):
        decoder_char_embeddings = tf.get_variable(
            "CharEmbeddings",
            [graph_part.chars_count, graph_part.settings['char_vector_size']],
            initializer=initializer
        )
        decoder_init_state = ClsGramEmbedder(graph_part.main_cls_dic,
                                          graph_part.settings['decoder']['gram_vector_size'],
                                          graph_part.settings['decoder']['ad_cls_vector_size'])(y_init, batch_size)


        #decoder_cls_emb = tf.get_variable(
        #    "YInitEmbeddings",
        #    (graph_part.main_classes_count, graph_part.settings['decoder']['rnn_state_size']),
        #    initializer=initializer
        #)
        #decoder_init_state = tf.nn.embedding_lookup(decoder_cls_emb, x_init)

        decoder_output = tf.nn.embedding_lookup(decoder_char_embeddings, y)

    if graph_part.for_usage:
        keep_drop = tf.constant(1, dtype=tf.float32, name='KeepDrop')
        decoder_keep_drop = tf.constant(1, dtype=tf.float32, name='DecoderKeepDrop')
    else:
        keep_drop = tf.placeholder(dtype=tf.float32, name='KeepDrop')
        decoder_keep_drop = tf.placeholder(dtype=tf.float32, name='DecoderKeepDrop')

    with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE) as scope:
        _, encoder_output = build_rnn(
            encoder_input,
            keep_drop,
            x_seq_len,
            graph_part.settings['encoder'],
            encoder_init_state,
            encoder_init_state,
            top_concat=False,
            for_usage=graph_part.for_usage,
            with_seq=True
        )

    with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE) as scope:
        if not graph_part.for_usage:
            start_tokens_emd = tf.nn.embedding_lookup(decoder_char_embeddings, start_tokens)
            start_tokens_emd = tf.reshape(start_tokens_emd, (batch_size, -1, graph_part.settings['char_vector_size']))
            decoder_output = tf.concat(values=[start_tokens_emd, decoder_output], axis=1)

            end_tokens = tf.fill([batch_size], graph_part.end_char_index)
            end_tokens_emd = tf.nn.embedding_lookup(decoder_char_embeddings, end_tokens)
            end_tokens_emd = tf.reshape(end_tokens_emd, (batch_size, -1, graph_part.settings['char_vector_size']))
            decoder_output = tf.concat([decoder_output, end_tokens_emd], axis=1)

            end_tokens = tf.reshape(end_tokens, (batch_size, 1))
            y = tf.concat([y, end_tokens], axis=1)

        if graph_part.for_usage:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_char_embeddings,
                                                              start_tokens=start_tokens,
                                                              end_token=graph_part.end_char_index)
        else:
            helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(decoder_output,
                                                                         y_seq_len,
                                                                         decoder_char_embeddings,
                                                                         graph_part.sampling_probability)

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=graph_part.settings['decoder']['rnn_state_size'],
            memory=encoder_output,
            memory_sequence_length=x_seq_len,
            normalize=False
        )

        cell = rnn_cell(graph_part.settings['decoder'],
                            graph_part.for_usage,
                            decoder_keep_drop)

        cell = tf.contrib.seq2seq.AttentionWrapper(
            cell,
            attention_mechanism,
            attention_layer_size=graph_part.settings['decoder']['rnn_state_size'] / 2
        )

        out_cell = tf.contrib.rnn.OutputProjectionWrapper(
            cell,
            graph_part.chars_count
        )

        init_state = cell.zero_state(dtype=tf.float32, batch_size=batch_size).clone(cell_state=decoder_init_state)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=out_cell,
            helper=helper,
            initial_state=init_state
        )

        max_len = graph_part.settings['max_length'] if graph_part.for_usage else tf.reduce_max(y_seq_len)
        outputs = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            impute_finished=True,
            output_time_major=False,
            maximum_iterations=max_len
        )
        decoder_ids = outputs[0].sample_id

        if not graph_part.for_usage:
            decoder_logits = outputs[0].rnn_output
            masks = tf.sequence_mask(
                lengths=y_seq_len,
                dtype=tf.float32,
                maxlen=tf.reduce_max(y_seq_len)
            )
            seq_mask_int = tf.cast(masks, tf.int32)
            seq_mask_flat = tf.cast(tf.reshape(masks, (-1,)), tf.int32)

            # seq loss
            loss = tf.contrib.seq2seq.sequence_loss(
                decoder_logits,
                y,
                masks,
                name="SeqLoss"
            )
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=graph_part.main_scope_name)
            grads = graph_part.optimiser.compute_gradients(loss, var_list=vars)
            graph_part.create_mean_metric(0, loss)

            labels_flat = tf.reshape(y, (-1,))
            predictions_flat = tf.reshape(decoder_ids, (-1,))
            # char accuracy
            nonzero_indices = tf.where(tf.not_equal(seq_mask_flat, 0))
            labels_flat = tf.gather(labels_flat, nonzero_indices)
            labels_flat = tf.reshape(labels_flat, (-1,))
            predictions_flat = tf.gather(predictions_flat, nonzero_indices)
            predictions_flat = tf.reshape(predictions_flat, (-1,))

            # remove -1 items where no sampling took place
            sample_indexes = tf.where(tf.not_equal(predictions_flat, -1))
            labels_flat = tf.gather(labels_flat, sample_indexes)
            predictions_flat = tf.gather(predictions_flat, sample_indexes)

            graph_part.create_accuracy_metric(1, labels_flat, predictions_flat)

            # seq accuracy
            labels = y * seq_mask_int
            predictions = decoder_ids * seq_mask_int
            labels_flat = tf.reshape(labels, (-1,))
            predictions_flat = tf.reshape(predictions, (-1,))
            sample_zeros = tf.cast(tf.not_equal(predictions_flat, -1), tf.int32)
            predictions = predictions_flat * sample_zeros
            labels = labels_flat * sample_zeros
            predictions = tf.reshape(predictions, (batch_size, max_len))
            labels = tf.reshape(labels, (batch_size, max_len))
            delta = labels - predictions
            labels = tf.reduce_sum(delta * delta, 1)
            predictions = tf.zeros(batch_size)
            graph_part.create_accuracy_metric(2, labels, predictions)

            graph_part.dev_grads.append(grads)
            graph_part.losses.append(loss)
            graph_part.keep_drops.append(keep_drop)
            graph_part.decoder_keep_drops.append(decoder_keep_drop)

        graph_part.results.append(decoder_ids)


class ClsGramEmbedder:
    def __init__(self, cls_dic, gram_vector_size, ad_cls_vector_size):
        gram_rez_dict = {}
        tpls = sorted([(key, cls_dic[key]) for key in cls_dic], key=lambda x: x[1])
        cls_vectors = []
        for cls_key, cls_index in tpls:
            cls_vector = []
            for gram, gram_index in enumerate(list(cls_key)):
                gram_key = (gram_index, gram)
                if gram_key not in gram_rez_dict:
                    gram_rez_dict[gram_key] = len(gram_rez_dict)

                cls_vector.append(gram_rez_dict[gram_key])

            cls_vectors.append(cls_vector)

        tpls = sorted([(key, gram_rez_dict[key]) for key in gram_rez_dict], key=lambda x: x[1])
        gram_vectors = []
        for gram_key, gram_index in tpls:
            if gram_key[0] is None:
                val = np.zeros(gram_vector_size, dtype=np.float32)
            else:
                val = np.random.rand(gram_vector_size).astype(np.float32)

            gram_vectors.append(val)

        self.classes_count = len(cls_dic)
        self.cls_vectors = np.asarray(cls_vectors, dtype=np.int)

        self.grams_count = len(gram_vectors)
        self.gram_vector_size = gram_vector_size
        self.gram_vectors = np.stack(gram_vectors)

        self.ad_cls_vector_size = ad_cls_vector_size

    def __call__(self, cls_pl, batch_size):
        # [self.grams_count, self.gram_vector_size],
        gram_embeddings = tf.get_variable(
            "GramEmbeddings",
            initializer=tf.constant(self.gram_vectors),
            dtype=tf.float32
        )
        # [self.classes_count, self.grams_count],
        cls_embeddings = tf.get_variable(
            "ClsEmbeddings",
            initializer=tf.constant(self.cls_vectors),
            dtype=tf.int64
        )

        gram_rez = tf.nn.embedding_lookup(cls_embeddings, cls_pl)
        gram_rez = tf.reshape(gram_rez, (-1, ))
        gram_rez = tf.nn.embedding_lookup(gram_embeddings, gram_rez)
        gram_rez = tf.reshape(gram_rez, (batch_size, -1))

        ad_cls_embeddings = tf.get_variable(
            "AdClsEmbeddings",
            [self.classes_count, self.ad_cls_vector_size],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        ad_cls_rez = tf.nn.embedding_lookup(ad_cls_embeddings, cls_pl)

        result = tf.concat([gram_rez, ad_cls_rez], axis=1)
        return result
