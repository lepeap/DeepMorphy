import pickle
import tensorflow as tf
import tf_utils as tfu
from graph.base import GraphPartBase
from utils import MAX_WORD_SIZE, load_tags


class Ambig(GraphPartBase):
    def __init__(self,
                 for_usage,
                 global_settings,
                 current_settings,
                 optimiser,
                 reset_optimiser):
        super().__init__(for_usage, global_settings, current_settings, optimiser, reset_optimiser, 'ambig', ['Loss', 'Accuracy'])
        self.tags = load_tags()
        self.tags_count = len(self.tags)
        self.main_classes_count = len(global_settings['main_classes'])
        self.amb_chars_count = len(global_settings['chars']) + len(global_settings['ambig_chars'])
        self.end_char = global_settings['end_token']

        self.main_drops = []
        self.gram_drops = []
        self.amb_x_inds = []
        self.amb_x_vals = []
        self.amb_x_shape = []
        self.xs_amb = []
        self.keep_drops = []

        self.sent_batch_sizes = []
        self.sent_max_lengths = []
        self.sent_lengths = []
        self.masks = []
        self.upper_masks = []
        self.ad_tags = []
        self.ys = []
        self.results = []
        self.checks = []
        self.losses = []


    def __build_graph_for_device__(self, x, x_seq_len, main_probs, seq_len, net_tags, gram_drop, main_drop):
        self.gram_drops.append(gram_drop)
        self.xs.append(x)
        self.x_seq_lens.append(x_seq_len)
        sent_batch_size = tf.placeholder(dtype=tf.int32, name='AmbigBatchSize')
        self.sent_batch_sizes.append(sent_batch_size)
        sent_max_length = tf.placeholder(dtype=tf.int32, name='SentMaxLen')
        self.sent_max_lengths.append(sent_max_length)
        sent_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='SentSeqLen')
        self.sent_lengths.append(sent_length)
        mask = tf.placeholder(dtype=tf.int32, shape=(None,), name='Masks')
        self.masks.append(mask)
        ad_tag = tf.placeholder(dtype=tf.int32, shape=(None, self.settings['ad_tags_max_count']), name='AdTags')
        self.ad_tags.append(ad_tag)
        y = tf.placeholder(dtype=tf.int64, shape=(None,), name='Y')
        self.ys.append(y)
        upper_mask = tf.placeholder(dtype=tf.int32, shape=(None, None, 1), name='UpperMask')
        self.upper_masks.append(upper_mask)
        self.main_drops.append(main_drop)

        if self.for_usage:
            keep_drop = tf.constant(1, dtype=tf.float32, name='KeepDrop')
        else:
            keep_drop = tf.placeholder(dtype=tf.float32, name='KeepDrop')
        self.keep_drops.append(keep_drop)

        #if self.for_usage:
        #    x_ind_pl = tf.placeholder(dtype=tf.int32, shape=(None, None), name='XIndexes')
        #    x_val_pl = tf.placeholder(dtype=tf.int32, shape=(None,), name='XValues')
        #    x_shape_pl = tf.placeholder(dtype=tf.int32, shape=(2,), name='XShape')
        #    x_ind = tf.dtypes.cast(x_ind_pl, dtype=tf.int64)
        #    x_val = tf.dtypes.cast(x_val_pl, dtype=tf.int64)
        #    x_shape = tf.dtypes.cast(x_shape_pl, dtype=tf.int64)
#
        #    x_sparse = tf.sparse.SparseTensor(x_ind, x_val, x_shape)
        #    x = tf.sparse.to_dense(x_sparse, default_value=self.end_char)
        #    self.amb_x_inds.append(x_ind_pl)
        #    self.amb_x_vals.append(x_val_pl)
        #    self.amb_x_shape.append(x_shape_pl)
        #else:
        x = tf.placeholder(dtype=tf.int32, shape=(None, None), name='X')
        self.xs_amb.append(x)

        main_probs = tf.reshape(main_probs, (sent_batch_size, sent_max_length, self.main_classes_count))
        word_length = tf.reshape(seq_len, (sent_batch_size, sent_max_length, 1))

        x_emd_init = tf.random_normal((self.amb_chars_count, self.settings['char_vector_size'] - 1))
        x_emb = tf.get_variable("Embeddings", initializer=x_emd_init)
        rnn_input = tf.nn.embedding_lookup(x_emb, x)
        rnn_input = tf.concat([rnn_input, tf.cast(upper_mask, dtype=tf.float32)], axis=2)
        with tf.variable_scope("WordRnn", reuse=tf.AUTO_REUSE) as scope:
            word_result = tfu.build_rnn(rnn_input,
                                        keep_drop,
                                        seq_len,
                                        self.settings['word_rnn'],
                                        for_usage=self.for_usage)
            word_result = tf.reshape(word_result, (sent_batch_size,
                                                   sent_max_length,
                                                   self.settings['word_rnn']['rnn_state_size']))

        tag_length = self.settings['ad_tags_max_count'] + net_tags.shape[-1]
        tag = tf.concat([net_tags, ad_tag], axis=1)
        tag = tf.reshape(tag, (-1,))
        tag = tf.reshape(tag, (sent_batch_size, sent_max_length, tag_length))

        rnn_input = tf.concat([
            word_result,
            main_probs,
            tf.cast(word_length, dtype=tf.float32),
            tf.cast(tag, dtype=tf.float32)],
            axis=2)

        _, rnn_logits = tfu.build_rnn(rnn_input,
                                      keep_drop,
                                      sent_length,
                                      self.settings,
                                      for_usage=self.for_usage,
                                      with_seq=True)
        rnn_result = tf.reshape(rnn_logits, (-1, rnn_logits.shape[-1].value))
        logits = tfu.rnn_top('RnnTop',
                             rnn_result,
                             self.settings,
                             self.tags_count)

        labels = tf.reshape(y, (-1,))
        probs = tf.nn.softmax(logits)
        errors = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=labels)

        #errors = errors * tf.cast(mask, tf.float32)
        if not self.for_usage:
            self.checks.append(tf.check_numerics(errors, "LossNullCheck"))

        loss = tf.reduce_sum(errors)
        self.losses.append(loss)

        result = tf.math.argmax(probs, axis=1)
        ac_result = tf.boolean_mask(result, tf.cast(mask, dtype=tf.bool))
        ac_labels = tf.boolean_mask(labels, tf.cast(mask, dtype=tf.bool))
        self.create_accuracy_metric(1, ac_labels, ac_result)
        self.create_mean_metric(0, loss)
        result = tf.reshape(result, (sent_batch_size, sent_max_length))
        self.results.append(result)

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.main_scope_name)
        grads = self.optimiser.compute_gradients(loss, var_list=vars)
        self.dev_grads.append(grads)

    def __update_feed_dict__(self, op_name, feed_dict, batch, dev_num):
        for gram_drop in self.gram_drops[dev_num]:
            feed_dict[gram_drop] = 1

        feed_dict[self.keep_drops[dev_num]] = self.settings['keep_drop'] if op_name == 'train' else 1
        feed_dict[self.main_drops[dev_num]] = 1
        feed_dict[self.xs_amb[dev_num]] = batch['x_amb']
        feed_dict[self.sent_max_lengths[dev_num]] = batch['sent_max_length']
        feed_dict[self.sent_lengths[dev_num]] = batch['sent_length']
        feed_dict[self.sent_batch_sizes[dev_num]] = batch['sent_batch_size']
        feed_dict[self.masks[dev_num]] = batch['mask']
        feed_dict[self.ad_tags[dev_num]] = batch['ad_tags']
        feed_dict[self.upper_masks[dev_num]] = batch['upper_mask']
        feed_dict[self.ys[dev_num]] = batch['y']

    def __load_dataset__(self, operation_name):
         return list(
            tfu.load_ambig_dataset(
                self.dataset_path,
                self.devices_count,
                operation_name,
                self.settings[f'{operation_name}_batch_size'],
                MAX_WORD_SIZE,
                self.settings['ad_tags_max_count']
            )
        )
