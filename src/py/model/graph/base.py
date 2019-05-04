import os
import tensorflow as tf
import tf_utils as tfu
from tqdm import tqdm
from abc import ABC, abstractmethod


class TrainContext:
    def __init__(self,
                 sess,
                 saver,
                 learn_rate_op):
        self.sess = sess
        self.saver = saver
        self.learn_rate_op = learn_rate_op
        self.epoch = 0


class GraphPartBase(ABC):
    def __init__(self,
                 for_usage,
                 global_settings,
                 current_settings,
                 optimiser,
                 key
                 ):
        self.key = key
        self.global_settings = global_settings
        self.filler = global_settings['filler']
        self.settings = current_settings
        self.for_usage = for_usage
        self.optimiser = optimiser
        self.metric_funcs = [
            ('Accuracy', tf.metrics.accuracy)
        ]
        self.max_word_size = global_settings['max_word_size']
        self.checkpoints_keep = 10000
        self.chars_count = len(global_settings['chars']) + 1
        self.dataset_path = global_settings['dataset_path']
        self.grammemes_count = len(global_settings['grammemes_types'])
        self.main_classes = global_settings['main_classes']
        self.main_classes_count = len(self.main_classes)
        self.metrics_reset = []
        self.metrics_update = []
        self.devices_metrics = {metr[0]: [] for metr in self.metric_funcs}
        self.main_scope_name = key.title()
        self.save_path = global_settings['save_path']
        self.dev_grads = []
        self.losses = []
        self.devices = global_settings['train_devices']
        self.devices_count = len(self.devices)
        self.dataset_path = global_settings['dataset_path']
        self.xs = []
        self.seq_lens = []

    def train(self, tc):
        best_model_acc = -1
        best_epoch = -1
        learn_rate_val = self.settings['learn_rate']
        acc_delta = 100
        return_step = 0
        trains = self.__load_dataset__('train')
        valids = self.__load_dataset__('valid')
        while self.settings['stop_training_acc_delta'] < acc_delta:
            tqdm.write(self.filler)
            tqdm.write(self.filler)
            tqdm.write(self.main_scope_name)
            tc.sess.run(self.metrics_reset)

            for item in tqdm(trains, desc=f"Train, epoch {tc.epoch}"):
                launch = [
                    self.optimize
                ]
                launch.extend(self.metrics_update)
                feed_dic = self.__create_feed_dict__('train', item)
                feed_dic[tc.learn_rate_op] = learn_rate_val
                tc.sess.run(launch, feed_dic)

            train_acc = self.__write_metrics_report__(tc.sess, "Train")
            tc.sess.run(self.metrics_reset)

            for item in tqdm(valids, desc=f"Validation, epoch {tc.epoch}"):
                launch = []
                launch.extend(self.metrics_update)
                feed_dic = self.__create_feed_dict__('valid', item)
                tc.sess.run(launch, feed_dic)

            valid_acc = self.__write_metrics_report__(tc.sess, "Valid")
            tc.sess.run(self.metrics_reset)

            tqdm.write(f"Epoch {tc.epoch} Train accuracy: {train_acc} Validation accuracy: {valid_acc}")
            need_decay = False
            if valid_acc > best_model_acc:
                if (valid_acc - best_model_acc) < self.settings['stop_training_acc_delta']:
                    tqdm.write(f"Acc delta is less then min value")
                    need_decay = True
                else:
                    return_step = 0

                best_model_acc = valid_acc
                best_epoch = tc.epoch
                tc.saver.save(tc.sess, self.save_path, tc.epoch)
                tc.epoch += 1
            else:
                tqdm.write("Best epoch is better then current")
                if return_step == self.settings['return_step']:
                    tqdm.write(f"Restoring best epoch {best_epoch}")
                    tc.saver.restore(tc.sess, os.path.join(self.save_path, f"-{best_epoch}"))
                need_decay = True

            if need_decay:
                if return_step == self.settings['return_step']:
                    learn_rate_val = learn_rate_val * self.settings['learn_rate_decay_step']
                    if learn_rate_val < self.settings['min_learn_rate']:
                        tqdm.write(f"Learning rate {learn_rate_val} is less then min learning rate")
                        break
                    tqdm.write(f"Learning rate decayed. New value: {learn_rate_val}")
                    return_step = 0
                else:
                    tqdm.write(f"Return step increased")
                    return_step += 1

        return best_epoch, best_model_acc

    def test(self, tc):
        tests = self.__load_dataset__('test')
        tc.sess.run(self.metrics_reset)
        for item in tqdm(tests, desc=f"Validation, epoch {tc.epoch}"):
            launch = []
            launch.extend(self.metrics_update)
            feed_dic = self.__create_feed_dict__('test', item)
            tc.sess.run(launch, feed_dic)

        self.__write_metrics_report__(tc.sess, f"Test {self.main_scope_name}")

    def build_graph_end(self):
        with tf.variable_scope(self.main_scope_name, reuse=tf.AUTO_REUSE) as scope:
            self.grads = tfu.average_gradients(self.dev_grads)
            self.optimize = self.optimiser.apply_gradients(self.grads, name='Optimize')
            self.loss = tf.reduce_sum(self.losses, name='GlobalLoss')
            self.metrics = {
                metr: tf.reduce_mean(self.devices_metrics[metr], name=metr)
                for metr in self.devices_metrics
            }

    def build_graph_for_device(self, *args):
        with tf.variable_scope(self.main_scope_name, reuse=tf.AUTO_REUSE) as scope:
            self.__build_graph_for_device__(*args)

    def __write_metrics_report__(self, sess, step_name):
        tqdm.write('')
        tqdm.write(f"{step_name}")
        launch_results = sess.run(self.metrics)
        result = []

        for metr in self.metrics:
            result.append('{:>8}'.format(metr))
            result.append("=")
            result.append("{0:.7f}".format(launch_results[metr]))
            result.append(" ")

        result = "".join(result)
        tqdm.write(result)
        return launch_results["Accuracy"]

    def __create_feed_dict__(self, op_name, item):
        feed_dic = {}
        for dev_num, batch in enumerate(item):
            feed_dic[self.xs[dev_num]] = batch['x']
            feed_dic[self.seq_lens[dev_num]] = batch['x_seq_len']
            self.__update_feed_dict__(op_name, feed_dic, batch, dev_num)

        return feed_dic

    @abstractmethod
    def __update_feed_dict__(self, op_name, feed_dict, batch, dev_num):
        pass

    @abstractmethod
    def __build_graph_for_device__(self, *args):
        pass

    @abstractmethod
    def __load_dataset__(self, operation_name):
        return []
