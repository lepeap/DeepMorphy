import os
import tensorflow as tf
import tf_utils as tfu
from tqdm import tqdm
from abc import ABC, abstractmethod
from utils import RANDOM


class TfContext:
    def __init__(self,
                 sess,
                 saver,
                 learn_rate_op):
        self.sess = sess
        self.saver = saver
        self.learn_rate_op = learn_rate_op
        self.epoch = 0


class GraphPartBase(ABC):
    def __init__(self, for_usage, global_settings, current_settings, optimiser, reset_optimiser, key, metric_names):
        self.key = key
        self.global_settings = global_settings
        self.filler = global_settings['filler']
        self.main_metric_name = current_settings['main_metric_type']
        self.settings = current_settings
        self.for_usage = for_usage
        self.optimiser = optimiser
        self.reset_optimiser = reset_optimiser
        self.metric_names = metric_names
        self.max_word_size = global_settings['max_word_size']
        self.checkpoints_keep = 10000
        self.chars_count = len(global_settings['chars']) + 1
        self.dataset_path = global_settings['dataset_path']
        self.grammemes_count = len(global_settings['grammemes_types'])
        self.main_classes = global_settings['main_classes']
        self.main_classes_count = len(self.main_classes)
        self.metrics_reset = []
        self.metrics_update = []
        self.devices_metrics = {metr: [] for metr in self.metric_names}
        self.main_scope_name = key.title()
        self.save_path = global_settings['save_path']
        self.dev_grads = []
        self.losses = []
        self.devices = global_settings['train_devices']
        self.devices_count = len(self.devices)
        self.dataset_path = global_settings['dataset_path']
        self.xs = []
        self.x_seq_lens = []
        self.prints = []
        self.main_cls_dic = self.global_settings['main_classes']
        self.learn_rate_val = self.settings['learn_rate']
        self.best_model_metric = None
        self.best_epoch = None
        self.init_checkpoint = None

    def train(self, tc):
        return_step = 0
        trains = self.__load_dataset__('train')
        valids = self.__load_dataset__('valid')
        self.best_model_metric = self.__valid_loop__(tc, valids)
        self.best_epoch = -1
        while True:
            tqdm.write(self.filler)
            tqdm.write(self.filler)
            tqdm.write(self.main_scope_name)

            train_main_metric = self.__train_loop__(tc, trains)
            valid_main_metric = self.__valid_loop__(tc, valids)

            tqdm.write(f"Epoch {tc.epoch} Train {self.main_metric_name}: {train_main_metric} Validation {self.main_metric_name}: {valid_main_metric}")
            need_decay = False
            delta = self.__calc_metric_delta__(self.best_model_metric, valid_main_metric)

            if delta > 0:
                if delta < self.settings['stop_main_metric_delta']:
                    tqdm.write(f"{self.main_metric_name} delta is less then min value")
                    need_decay = True
                else:
                    return_step = 0
                self.best_model_metric = valid_main_metric
                self.best_epoch = tc.epoch
                tc.saver.save(tc.sess, self.save_path, tc.epoch)
                tc.epoch += 1
            else:
                tqdm.write("Best epoch is better then current")
                tc.sess.run(self.reset_optimiser)
                need_decay = True
                self.__restore_best_epoch__(tc)

            if not need_decay:
                continue

            if return_step == self.settings['return_step']:
                self.__decay_params__()
                if self.learn_rate_val < self.settings['min_learn_rate']:
                    tqdm.write(f"Learning rate {self.learn_rate_val} is less then min learning rate")
                    finish = self.__before_finish__()
                    if finish:
                        break
                return_step = 0
            else:
                RANDOM.shuffle(trains)
                tqdm.write(f"Return step increased")
                return_step += 1

        return self.best_epoch, self.best_model_metric

    def __train_loop__(self, tc, trains):
        tc.sess.run(self.metrics_reset)
        for item in tqdm(trains, desc=f"Train, epoch {tc.epoch}"):
            launch = [self.optimize]
            launch.extend(self.metrics_update)
            if len(self.prints):
                launch.extend(self.prints)

            feed_dic = self.__create_feed_dict__('train', item)
            feed_dic[tc.learn_rate_op] = self.learn_rate_val
            tc.sess.run(launch, feed_dic)

        train_main_metric = self.__write_metrics_report__(tc.sess, "Train")
        return train_main_metric

    def __valid_loop__(self, tc, valids):
        tc.sess.run(self.metrics_reset)
        for item in tqdm(valids, desc=f"Validation, epoch {tc.epoch}"):
            launch = []
            launch.extend(self.metrics_update)
            if len(self.prints):
                launch.extend(self.prints)
            feed_dic = self.__create_feed_dict__('valid', item)
            tc.sess.run(launch, feed_dic)

        valid_main_metric = self.__write_metrics_report__(tc.sess, "Valid")
        return valid_main_metric

    def __calc_metric_delta__(self, best_model_metric, cur_model_metric):
        delta = best_model_metric - cur_model_metric
        if self.main_metric_name != "Loss":
            delta = -delta
        return delta

    def __before_finish__(self):
        return True

    def __decay_params__(self):
        self.learn_rate_val = self.learn_rate_val * self.settings['learn_rate_decay_step']
        tqdm.write(f"Learning rate decayed. New value: {self.learn_rate_val}")

    def __restore_best_epoch__(self, tc):
        if self.best_epoch == -1 and tc.epoch == 0:
            tqdm.write(f"Restoring from init_checkpoint {self.best_epoch}")
            self.restore(tc.sess, self.init_checkpoint)
        elif self.best_epoch == -1:
            tqdm.write(f"Restoring best epoch {tc.epoch}")
            self.restore(tc.sess, os.path.join(self.save_path, f"-{tc.epoch}"))
        else:
            tqdm.write(f"Restoring best epoch {self.best_epoch}")
            self.restore(tc.sess, os.path.join(self.save_path, f"-{self.best_epoch}"))

    def build_graph_end(self):
        with tf.variable_scope(self.main_scope_name, reuse=tf.AUTO_REUSE) as scope:
            self.metrics = {
                metr: tf.reduce_mean(self.devices_metrics[metr], name=metr)
                for metr in self.devices_metrics
            }
            if not self.for_usage:
                self.grads = tfu.average_gradients(self.dev_grads)
                if self.settings['clip_grads']:
                    self.grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.grads]

                self.optimize = self.optimiser.apply_gradients(self.grads, name='Optimize')
                self.loss = tf.reduce_sum(self.losses, name='GlobalLoss')

    def build_graph_for_device(self, *args):
        with tf.variable_scope(self.main_scope_name, reuse=tf.AUTO_REUSE) as scope:
            self.__build_graph_for_device__(*args)

    def restore(self, sess, check_point):
        try:
            vars = [
                var
                for var in tf.global_variables(f"{self.main_scope_name}/")
                if "Adam" not in var.name
            ]
            saver = tf.train.Saver(var_list=vars)
            saver.restore(sess, check_point)
            self.init_checkpoint = check_point
            tqdm.write(f"Restoration for graph part '{self.key}', scope {self.main_scope_name} success")
        except Exception as ex:
            tqdm.write(f"Restoration for graph part '{self.key}', scope {self.main_scope_name} failed. Error: {ex}")

    def __write_metrics_report__(self, sess, step_name):
        tqdm.write('')
        launch_results = sess.run(self.metrics)
        result = [f"{step_name} metrics: "]

        for index, metr in enumerate(self.metrics):
            result.append('{:>8}'.format(self.metric_names[index]))
            result.append("=")
            result.append("{0:.7f}".format(launch_results[metr]))
            result.append(" ")

        result = "".join(result)
        tqdm.write(result)
        return launch_results[self.main_metric_name]

    def create_mean_metric(self, metric_index, values):
        metr_epoch_loss, metr_update, metr_reset = tfu.create_reset_metric(
            tf.metrics.mean,
            self.metric_names[metric_index],
            values
        )
        self.metrics_reset.append(metr_reset)
        self.metrics_update.append(metr_update)
        self.devices_metrics[self.metric_names[metric_index]].append(metr_epoch_loss)

    def create_accuracy_metric(self, metric_index, labels, predictions):
        metr_epoch_loss, metr_update, metr_reset = tfu.create_reset_metric(
            tf.metrics.accuracy,
            self.metric_names[metric_index],
            labels=labels,
            predictions=predictions
        )
        self.metrics_reset.append(metr_reset)
        self.metrics_update.append(metr_update)
        self.devices_metrics[self.metric_names[metric_index]].append(metr_epoch_loss)

    def __create_feed_dict__(self, op_name, item):
        feed_dic = {}
        for dev_num, batch in enumerate(item):
            feed_dic[self.xs[dev_num]] = batch['x']
            feed_dic[self.x_seq_lens[dev_num]] = batch['x_seq_len']
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
