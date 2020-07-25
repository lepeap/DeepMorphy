
import tensorflow as tf
import tf_utils as tfu
from tqdm import tqdm
from graph.base import GraphPartBase


class Lemm(GraphPartBase):
    def __init__(self, for_usage, global_settings, current_settings, optimiser):
        super().__init__(for_usage, global_settings, current_settings, optimiser, 'lemm', ["Loss", "AccuracyByChar", "Accuracy"])
        self.chars_count = self.chars_count + 1
        self.start_char_index = global_settings['start_token']
        self.end_char_index = global_settings['end_token']
        self.results = []
        self.ys = []
        self.y_seq_lens = []
        self.cls = []
        self.keep_drops = []
        self.decoder_keep_drops = []

        self.sampling_probability = None
        self.use_sampling = current_settings['use_sampling']
        self.sampling_probability_value = current_settings['sampling_probability']
        self.use_cls_placeholder = self.settings['use_cls_placeholder']

    def __build_graph_for_device__(self, x, x_seq_len, batch_size, x_cls=None):
        self.xs.append(x)
        self.x_seq_lens.append(x_seq_len)

        if x_cls is None or self.use_cls_placeholder:
            x_cls = tf.placeholder(dtype=tf.int32, shape=(None,), name='XClass')
            self.cls.append(x_cls)

        if batch_size is None:
            batch_size = self.settings['batch_size']

        y = tf.placeholder(dtype=tf.int32, shape=(None, None), name='Y')
        y_seq_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='YSeqLen')
        if self.use_sampling and self.sampling_probability is None:
            self.sampling_probability = tf.placeholder(tf.float32, [], name='SamplingProbability')

        self.ys.append(y)
        self.y_seq_lens.append(y_seq_len)

        tfu.seq2seq(self,
                    batch_size,
                    x,
                    x_cls,
                    x_seq_len,
                    y,
                    x_cls,
                    y_seq_len)

    def __update_feed_dict__(self, op_name, feed_dict, batch, dev_num):
        feed_dict[self.cls[dev_num]] = batch['x_cls']
        feed_dict[self.ys[dev_num]] = batch['y']
        feed_dict[self.y_seq_lens[dev_num]] = batch['y_seq_len']
        feed_dict[self.keep_drops[dev_num]] = self.settings['keep_drop']
        feed_dict[self.decoder_keep_drops[dev_num]] = self.settings['decoder']['keep_drop']
        if self.use_sampling:
            feed_dict[self.sampling_probability] = self.sampling_probability_value

    def __load_dataset__(self, operation_name):
        items = list(tfu.load_lemma_dataset(
            self.dataset_path,
            self.devices_count,
            operation_name,
            self.settings['batch_size']
        ))
        return items
