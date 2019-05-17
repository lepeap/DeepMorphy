import tensorflow as tf
import tf_utils as tfu
from graph.base import GraphPartBase


class GramCls(GraphPartBase):
    def __init__(self,
                 key,
                 for_usage,
                 global_settings,
                 current_settings,
                 optimiser):
        super().__init__(for_usage,
                         global_settings,
                         current_settings,
                         optimiser,
                         key,
                         ['Loss', 'Accuracy']
                         )
        self.gram = key
        self.grammemes = global_settings['grammemes_types']
        self.classes = self.grammemes[key]['classes']
        self.classes_count = len(self.classes)
        self.checks = []
        self.weights = []
        self.keep_drops = []
        self.dev_grads = []
        self.probs = []
        self.results = []
        self.ys = []
        self.losses = []

    def __build_graph_for_device__(self, x, seq_len):
        self.xs.append(x)
        self.seq_lens.append(seq_len)

        if self.for_usage:
            keep_drop = tf.constant(1, dtype=tf.float32, name='KeepDrop')
        else:
            keep_drop = tf.placeholder(dtype=tf.float32, name='KeepDrop')

        self.keep_drops.append(keep_drop)

        y = tf.placeholder(dtype=tf.int32,
                                    shape=(None, self.classes_count),
                                    name='Y')
        weights = tf.placeholder(dtype=tf.float32, shape=(None,), name='Weight')

        x_emd_init = tf.random_normal((self.chars_count, self.settings['char_vector_size']))
        x_emb = tf.get_variable("Embeddings", initializer=x_emd_init)

        rnn_input = tf.nn.embedding_lookup(x_emb, x)
        rnn_logits = tfu.build_rnn(rnn_input,
                                   keep_drop,
                                   seq_len,
                                   self.settings,
                                   for_usage=self.for_usage)

        if not self.for_usage:
            self.checks.append(tf.check_numerics(rnn_logits, "RnnLogitsNullCheck"))

        logits = tfu.rnn_top('RnnTop',
                                      rnn_logits,
                                      self.settings,
                                      self.classes_count)

        if not self.for_usage:
            self.checks.append(tf.check_numerics(logits, "LogitsNullCheck"))

        float_y = tf.cast(y, tf.float32)
        errors = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                         labels=float_y)
        errors = tf.reduce_sum(errors, axis=1)
        errors = errors * weights

        if not self.for_usage:
            self.checks.append(tf.check_numerics(errors, "ErrorNullCheck"))

        probs = tf.nn.softmax(logits)
        result = tf.argmax(probs, axis=1, name="Results")
        loss = tf.reduce_sum(errors)

        if not self.for_usage:
            self.checks.append(tf.check_numerics(errors, "LossNullCheck"))

        grads = self.optimiser.compute_gradients(loss)
        self.losses.append(loss)
        self.probs.append(probs)
        self.results.append(result)
        self.ys.append(y)
        self.weights.append(weights)
        self.dev_grads.append(grads)

        # metrics
        self.__create_mean_metric__(0, loss)
        labels = tf.math.argmax(y, axis=1)
        predictions = tf.math.argmax(probs, axis=1)
        self.__create_accuracy_metric__(1, labels, predictions)

    def __update_feed_dict__(self, op_name, feed_dict, batch, dev_num):
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
                self.settings['use_weights'],
                self.gram
            )
        )


