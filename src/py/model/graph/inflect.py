import tensorflow as tf
import tf_utils as tfu
from graph.base import GraphPartBase


class Inflect(GraphPartBase):
    def __init__(self, for_usage, global_settings, current_settings, optimiser, reset_optimiser):
        super().__init__(for_usage, global_settings, current_settings, optimiser, reset_optimiser, 'inflect', ["Loss", "AccuracyByChar", "Accuracy"])
        self.chars_count = self.chars_count + 1
        self.start_char_index = global_settings['start_token']
        self.end_char_index = global_settings['end_token']
        self.results = []
        self.x_cls = []
        self.ys = []
        self.y_seq_lens = []
        self.y_cls = []
        self.keep_drops = []
        self.decoder_keep_drops = []
        self.sampling_probability = current_settings['sampling_probability']

    def __build_graph_for_device__(self, x, x_seq_len, batch_size, cls=None):
        self.xs.append(x)
        self.x_seq_lens.append(x_seq_len)

        x_cls = tf.placeholder(dtype=tf.int32, shape=(None,), name='XClass')
        self.x_cls.append(x_cls)

        if batch_size is None:
            batch_size = self.settings['batch_size']

        y = tf.placeholder(dtype=tf.int32, shape=(None, None), name='Y')
        self.ys.append(y)

        y_seq_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='YSeqLen')
        self.y_seq_lens.append(y_seq_len)

        y_cls = tf.placeholder(dtype=tf.int32, shape=(None,), name='YClass')
        self.y_cls.append(y_cls)

        tfu.seq2seq(self,
                    batch_size,
                    x,
                    y_cls,
                    x_seq_len,
                    y,
                    x_cls,
                    y_seq_len)

    def __update_feed_dict__(self, op_name, feed_dict, batch, dev_num):
        feed_dict[self.x_cls[dev_num]] = batch['x_cls']
        feed_dict[self.y_cls[dev_num]] = batch['y_cls']
        feed_dict[self.ys[dev_num]] = batch['y']
        feed_dict[self.y_seq_lens[dev_num]] = batch['y_seq_len']
        feed_dict[self.keep_drops[dev_num]] = self.settings['keep_drop']
        feed_dict[self.decoder_keep_drops[dev_num]] = self.settings['decoder']['keep_drop']

    def __load_dataset__(self, operation_name):
        items = list(tfu.load_inflect_dataset(
            self.dataset_path,
            self.devices_count,
            operation_name,
            self.settings['batch_size']
        ))
        return items
