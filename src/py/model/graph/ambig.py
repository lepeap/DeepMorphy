import pickle
import tensorflow as tf
import tf_utils as tfu
from graph.base import GraphPartBase


class Ambig(GraphPartBase):
    def __init__(self,
                 for_usage,
                 global_settings,
                 current_settings,
                 optimiser,
                 reset_optimiser):
        super().__init__(for_usage, global_settings, current_settings, optimiser, reset_optimiser, 'main', ['Loss', 'Accuracy'])
        with open(global_settings['tags_path'], 'rb') as f:
            self.tags = pickle.load(f)

        self.classes_count = len(self.tags)
        self.amb_chars_count = len(global_settings['chars']) + len(global_settings['ambig_chars'])
        self.end_char = global_settings['end_token']

        self.amb_x_inds = []
        self.amb_x_vals = []
        self.amb_x_shape = []
        self.amb_xs = []

        self.sent_batch_sizes = []
        self.sent_max_lengths = []
        self.weights = []
        self.word_lengths = []
        self.ad_tags = []

        self.checks = []

    def __build_graph_for_device__(self, main_probs, seq_len):
        sent_batch_size = tf.placeholder(dtype=tf.int32, name='SentBatchSize')
        sent_max_length = tf.placeholder(dtype=tf.int32, name='SentMaxLen')
        sent_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='SentSeqLen')
        weight = tf.placeholder(dtype=tf.int32, shape=(None,), name='Weights')
        word_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='WordLength')
        ad_tag = tf.placeholder(dtype=tf.int32, shape=(None, self.settings['ad_tags_max_count']), name='AdTags')
        if self.for_usage:
            x_ind_pl = tf.placeholder(dtype=tf.int32, shape=(None, None), name='XIndexes')
            x_val_pl = tf.placeholder(dtype=tf.int32, shape=(None,), name='XValues')
            x_shape_pl = tf.placeholder(dtype=tf.int32, shape=(2,), name='XShape')
            x_ind = tf.dtypes.cast(x_ind_pl, dtype=tf.int64)
            x_val = tf.dtypes.cast(x_val_pl, dtype=tf.int64)
            x_shape = tf.dtypes.cast(x_shape_pl, dtype=tf.int64)

            x_sparse = tf.sparse.SparseTensor(x_ind, x_val, x_shape)
            x = tf.sparse.to_dense(x_sparse, default_value=self.end_char)
            self.amb_x_inds.append(x_ind_pl)
            self.amb_x_vals.append(x_val_pl)
            self.amb_x_shape.append(x_shape_pl)
        else:
            x = tf.placeholder(dtype=tf.int32, shape=(None, None), name='X')
            self.amb_xs.append(x)

        x_emd_init = tf.random_normal((self.amb_chars_count, self.settings['char_vector_size']))
        x_emb = tf.get_variable("Embeddings", initializer=x_emd_init)
        rnn_input = tf.nn.embedding_lookup(x_emb, x)
        word_result = tfu.build_rnn(rnn_input, 1, seq_len, self.settings['word_rnn'], for_usage=self.for_usage)
        word_result = tf.reshape(word_result, (sent_batch_size, sent_max_length, -1))

        main_probs = tf.reshape(main_probs, (sent_batch_size, sent_max_length, -1))
        word_length = tf.reshape(word_length, (sent_batch_size, sent_max_length, 1))
        ad_tag = tf.reshape(ad_tag, (sent_batch_size, sent_max_length, self.settings['ad_tags_max_count']))
        rnn_input = tf.concat([word_result, main_probs, word_length, ad_tag])
        _, rnn_logits = tfu.build_rnn(rnn_input,
                                      1,
                                      sent_length,
                                      self.settings,
                                      for_usage=self.for_usage,
                                      with_seq=True)

        logits = tfu.rnn_top('RnnTop',
                             rnn_logits,
                             self.settings,
                             self.classes_count)

        float_y = tf.cast(y, tf.float32)
        errors = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=float_y)
        errors = errors * weight
        loss = tf.reduce_sum(errors)
        probs = tf.nn.softmax(logits)
        result = tf.math.argmax(probs)
        result = tf.reshape(result, (sent_batch_size, sent_max_length))
        if not self.for_usage:
            self.checks.append(tf.check_numerics(errors, "LossNullCheck"))

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.main_scope_name)
        grads = self.optimiser.compute_gradients(loss, var_list=vars)


        self.sent_batch_sizes.append(sent_batch_size)
        self.sent_max_lengths.append(sent_max_length)

    def __update_feed_dict__(self, op_name, feed_dict, batch, dev_num):
        for gram_drop in self.drops[dev_num]:
            feed_dict[gram_drop] = 1
        feed_dict[self.keep_drops[dev_num]] = 1 if op_name == 'test' else self.settings['keep_drop']
        feed_dict[self.ys[dev_num]] = batch['y']
        feed_dict[self.weights[dev_num]] = batch['weight']

    def __load_dataset__(self, operation_name):
        return list(
            tfu.load_cls_dataset(
                self.dataset_path,
                self.devices_count,
                operation_name,
                self.settings['train_batch_size'],
                self.settings['use_weights']
            )
        )
