import tensorflow as tf
import tf_utils as tfu
from graph.base import GraphPartBase
from tensorflow.python.ops.ragged.ragged_util import repeat


class Lemm(GraphPartBase):

    def __init__(self, for_usage, global_settings, current_settings, optimiser):
        super().__init__(for_usage, global_settings, current_settings, optimiser, 'lemm')
        self.chars_count = self.chars_count + 2
        self.start_char_index = global_settings['start_token']
        self.end_char_index = global_settings['end_token']
        self.results = []
        self.ys = []
        self.y_seq_lens = []
        self.cls = []

    def __build_graph_for_device__(self, x, seq_len, batch_size, cls):
        self.xs.append(x)
        self.seq_lens.append(seq_len)

        cls = tf.placeholder(dtype=tf.int32, shape=(None,), name='XClass')
        y = tf.placeholder(dtype=tf.int32, shape=(None, None), name='Y')
        y_seq_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='YSeqLen')

        x_emd_init = tf.random_normal((self.chars_count, self.settings['char_vector_size']))
        x_emb = tf.get_variable("EmbeddingsX", initializer=x_emd_init)
        encoder_input = tf.nn.embedding_lookup(x_emb, x)

        lemma_cls_init = tf.random_normal((self.main_classes_count, self.settings['encoder']['rnn_state_size']))
        lemma_cls_emb = tf.get_variable("EmbeddingsLemmaCls", initializer=lemma_cls_init)
        init_state = tf.nn.embedding_lookup(lemma_cls_emb, cls)

        y_emd_init = tf.random_normal((self.chars_count, self.settings['char_vector_size']))
        y_emb = tf.get_variable("EmbeddingsY", initializer=y_emd_init)
        lemma_output = tf.nn.embedding_lookup(y_emb, y)

        if self.for_usage:
            keep_drop = tf.constant(1, dtype=tf.float32, name='KeepDrop')
        else:
            keep_drop = tf.placeholder(dtype=tf.float32, name='KeepDrop')

        with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE) as scope:
            encoder_output = tfu.build_rnn(
                encoder_input,
                keep_drop,
                seq_len,
                self.settings['encoder'],
                init_state,
                init_state,
                self.for_usage
            )

        with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE) as scope:

            if self.for_usage:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(encoder_output,
                                                                  start_tokens=tf.fill([batch_size], self.start_char_index),
                                                                  end_token=self.end_char_index)
            else:
                helper = tf.contrib.seq2seq.TrainingHelper(lemma_output,
                                                           [self.settings['max_length'] for _ in range(batch_size)])
            decoder_output = tfu.decoder(
                helper,
                encoder_output,
                seq_len,
                self.settings,
                self.chars_count,
                batch_size,
                self.for_usage
            )


        masks = tf.sequence_mask(
            lengths=y_seq_len,
            dtype=tf.float32,
            maxlen=self.settings['max_length']
        )
        loss = tf.contrib.seq2seq.sequence_loss(decoder_output.rnn_output, y, masks)
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.main_scope_name)
        grads = self.optimiser.compute_gradients(loss, var_list=vars)

        for metric_func in self.metric_funcs:
            seq_mask = tf.cast(tf.reshape(masks, (-1,)), tf.int32)
            nonzero_indices = tf.where(tf.not_equal(seq_mask, 0))
            labels = tf.reshape(y, (-1,))
            labels = tf.gather(labels, nonzero_indices)
            predictions = tf.reshape(decoder_output.sample_id, (-1,))
            predictions = tf.gather(predictions, nonzero_indices)

            metr_epoch_loss, metr_update, metr_reset = tfu.create_reset_metric(
                metric_func[1],
                metric_func[0],
                labels=labels,
                predictions=predictions
            )
            self.metrics_reset.append(metr_reset)
            self.metrics_update.append(metr_update)
            self.devices_metrics[metric_func[0]].append(metr_epoch_loss)

        self.results.append(decoder_output.sample_id)
        self.ys.append(y)
        self.y_seq_lens.append(y_seq_len)
        self.cls.append(cls)
        self.dev_grads.append(grads)
        self.losses.append(loss)

    def __update_feed_dict__(self, op_name, feed_dict, batch, dev_num):
        feed_dict[self.cls[dev_num]] = batch['x_cls']
        feed_dict[self.ys[dev_num]] = batch['y']
        feed_dict[self.y_seq_lens[dev_num]] = batch['y_seq_len']

    def __load_dataset__(self, operation_name):
        return list(
            tfu.load_lemma_dataset(
                self.dataset_path,
                self.devices_count,
                operation_name,
                self.settings['batch_size']
            )
        )


