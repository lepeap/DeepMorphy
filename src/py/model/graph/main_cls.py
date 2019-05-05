import tensorflow as tf
import tf_utils as tfu
from graph.base import GraphPartBase


class MainCls(GraphPartBase):
    def __init__(self, for_usage,
                       global_settings,
                       current_settings,
                       optimiser
                 ):
        super().__init__(for_usage, global_settings, current_settings, optimiser, 'main')
        self.checks = []
        self.weights = []
        self.keep_drops = []
        self.losses = []
        self.probs = []
        self.results = []
        self.ys = []
        self.drops = []
        self.top_k = global_settings['main_class_k']

    def __build_graph_for_device__(self, x, seq_len, gram_probs, gram_drop):
        self.xs.append(x)
        self.seq_lens.append(seq_len)
        self.drops.append(gram_drop)

        y = tf.placeholder(dtype=tf.int32, shape=(None, self.main_classes_count), name='Y')
        weights = tf.placeholder(dtype=tf.float32, shape=(None,), name='Weight')

        x_emd_init = tf.random_normal((self.chars_count, self.settings['char_vector_size']))
        x_emb = tf.get_variable("Embeddings", initializer=x_emd_init)
        rnn_input = tf.nn.embedding_lookup(x_emb, x)

        if self.for_usage:
            cls_keep_drop = tf.constant(1, dtype=tf.float32, name='KeepDrop')
        else:
            cls_keep_drop = tf.placeholder(dtype=tf.float32, name='KeepDrop')
        self.keep_drops.append(cls_keep_drop)

        init_state = tf.concat(gram_probs, 1)

        with tf.variable_scope("InitRnnState", reuse=tf.AUTO_REUSE) as scope:
            rez_size = self.settings['rnn_state_size']
            w_softmax = tf.get_variable("W", (init_state.shape[1], rez_size))
            b_softmax = tf.get_variable("b", [rez_size])
            init_state = tf.matmul(init_state, w_softmax) + b_softmax

        rnn_logits = tfu.build_rnn(rnn_input,
                                   cls_keep_drop,
                                   seq_len,
                                   self.settings,
                                   init_state,
                                   init_state,
                                   self.for_usage)

        logits = tfu.rnn_top('RnnTop',
                             rnn_logits,
                             self.settings,
                             self.main_classes_count)

        float_y = tf.cast(y, tf.float32)
        errors = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                         labels=float_y)
        errors = tf.reduce_sum(errors, axis=1)
        errors = errors * weights
        if not self.for_usage:
            self.checks.append(tf.check_numerics(errors, "ErrorNullCheck"))

        probs = tf.nn.softmax(logits)
        result = tf.math.top_k(probs, self.top_k, name="Results")
        loss = tf.reduce_sum(errors)

        if not self.for_usage:
            self.checks.append(tf.check_numerics(errors, "LossNullCheck"))

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.main_scope_name)
        grads = self.optimiser.compute_gradients(loss, var_list=vars)

        self.losses.append(loss)
        self.probs.append(probs)
        self.results.append(result)
        self.ys.append(y)
        self.dev_grads.append(grads)
        self.weights.append(weights)

        for metric_func in self.metric_funcs:
            labels = tf.math.argmax(y, axis=1)
            predictions = tf.math.argmax(probs, axis=1)
            metr_epoch_loss, metr_update, metr_reset = tfu.create_reset_metric(
                metric_func[1],
                metric_func[0],
                labels=labels,
                predictions=predictions
            )
            self.metrics_reset.append(metr_reset)
            self.metrics_update.append(metr_update)
            self.devices_metrics[metric_func[0]].append(metr_epoch_loss)

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