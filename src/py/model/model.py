import os
import yaml
import pickle
import shutil
import tensorflow as tf
import tf_utils as tfu
from tqdm import tqdm
from utils import MyDefaultDict
from collections import OrderedDict
from tensorflow.python.tools import freeze_graph

class RNN:
    def __init__(self, for_usage):
        with open('config.yml', 'r') as f:
            self.config = yaml.load(f)
        self._for_usage = for_usage
        self._default_cls_gram_settings = self.config['gram_classification']['default']
        self._cls_gram_settings = MyDefaultDict(
            lambda key: self._default_cls_gram_settings,
            {
                key: MyDefaultDict(lambda key: self._default_cls_gram_settings[key], self.config['morph_classification'][key])
                for key in self.config['gram_classification']
                if key != 'default'
            }
        )
        self._cls_settings = self.config['classification']
        self._max_word_size = self.config['max_word_size']
        self._checkpoints_keep = 10000
        self._chars_count = len(self.config['chars'])+1
        self._dataset_path = self.config['dataset_path']
        self._grammemes = self.config['grammemes_types']
        self._grammemes_count = len(self.config['grammemes_types'])
        self._export_path = self.config['export_path']
        self._save_path = self.config['save_path']
        self._publish_path = self.config['publish_net_path']
        self._model_key = self.config['model_key']
        self._miss_steps = self.config['miss_steps'] if 'miss_steps' in self.config else []
        self._filler = "############################################################################"

        with open(os.path.join(self._dataset_path, f"classification_classes.pkl"), 'rb') as f:
            self._classes_dic = pickle.load(f)
            self._classes_count = len(self._classes_dic)

        if for_usage:
            self._devices = ['/cpu:0']
        else:
            self._devices = ['/gpu:0','/gpu:1','/gpu:2']

        self._devices_count = len(self._devices)
        if not os.path.exists(self._save_path):
            os.mkdir(self._save_path)

        self._build_graph()


    def _build_graph(self):
        self.graph = tf.Graph()
        self.checks = []
        self.xs = []
        self.seq_lens = []
        self.metric_funcs = [
            ('Accuracy', tf.metrics.accuracy),
            ('Recall', tf.metrics.recall),
            ('Precision', tf.metrics.precision)
        ]

        self.cls_gram_weights = {key: [] for key in self._grammemes}
        self.cls_gram_keep_drops = {key:[] for key in self._grammemes}
        self.cls_gram_dev_grads = {key:[] for key in self._grammemes}
        self.cls_gram_probs = {key:[] for key in self._grammemes}
        self.cls_gram_results = {key:[] for key in self._grammemes}
        self.cls_gram_ys = {key:[] for key in self._grammemes}
        self.cls_gram_losses = {key:[] for key in self._grammemes}
        self.cls_gram_metrics_reset = {key:[] for key in self._grammemes}
        self.cls_gram_metrics_update = {key:[] for key in self._grammemes}
        self.cls_gram_devices_metrics = OrderedDict({
            cls: {metr[0]: [] for metr in self.metric_funcs}
            for cls in self._grammemes
        })


        self.cls_top_ks = []
        self.cls_weights = []
        self.cls_keep_drops = []
        self.cls_losses = []
        self.cls_probs = []
        self.cls_results = []
        self.cls_ys = []
        self.cls_dev_grads = []
        self.cls_metrics_reset = []
        self.cls_metrics_update = []
        self.cls_devices_metrics = {metr[0]: [] for metr in self.metric_funcs}



        self.lem_ys = []




        with self.graph.as_default(), tf.device('/cpu:0'):
            self.is_training = tf.placeholder(tf.bool, name="IsTraining")
            self.cls_learn_rate = tf.placeholder(tf.float32, name="LearningRate")
            optimiser = tf.train.AdamOptimizer(self.cls_learn_rate)
            for deviceIndex, deviceName in enumerate(self._devices):
                with tf.device(deviceName):
                    x = tf.placeholder(dtype=tf.int32, shape=(None, None), name='X')
                    seq_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='SeqLen')
                    self.seq_lens.append(seq_len)
                    self.xs.append(x)

                    with tf.variable_scope('GramClassification', reuse=tf.AUTO_REUSE) as scope:
                        for gram in self._grammemes:
                            with tf.variable_scope(gram, reuse=tf.AUTO_REUSE) as scope:
                                if self._for_usage:
                                    cls_gram_keep_drop = tf.constant(1, dtype=tf.float32, name='KeepDrop')
                                else:
                                    cls_gram_keep_drop = tf.placeholder(dtype=tf.float32, name='KeepDrop')

                                self.cls_gram_keep_drops[gram].append(cls_gram_keep_drop)

                                cls_gram_y = tf.placeholder(dtype=tf.int32, shape=(None,), name='Y')
                                cls_gram_weights = tf.placeholder(dtype=tf.float32, shape=(None,), name='Weight')

                                settings = self._cls_gram_settings[gram]
                                x_emd_init = tf.random_normal((self._chars_count, settings['char_vector_size']))
                                x_emb = tf.get_variable("Embeddings", initializer=x_emd_init)

                                rnn_input = tf.nn.embedding_lookup(x_emb, x)
                                rnn_logits = tfu.build_rnn(rnn_input, cls_gram_keep_drop, seq_len, settings, for_usage=self._for_usage)

                                if not self._for_usage:
                                    self.checks.append(tf.check_numerics(rnn_logits, "RnnLogitsNullCheck"))

                                classes_number = len(self._grammemes[gram]['classes'])
                                cls_gram_logits = tfu.rnn_top('RnnTop', rnn_logits, settings, classes_number)

                                if not self._for_usage:
                                    self.checks.append(tf.check_numerics(cls_gram_logits, "LogitsNullCheck"))

                                cls_gram_errors = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_gram_logits, labels=cls_gram_y)
                                cls_gram_errors = cls_gram_errors * cls_gram_errors

                                if not self._for_usage:
                                    self.checks.append(tf.check_numerics(cls_gram_errors, "ErrorNullCheck"))

                                cls_gram_probs = tf.nn.softmax(cls_gram_logits)
                                cls_gram_result = tf.argmax(cls_gram_probs, axis=1, name="Results")
                                cls_gram_loss = tf.reduce_sum(cls_gram_errors)

                                if not self._for_usage:
                                    self.checks.append(tf.check_numerics(cls_gram_errors, "LossNullCheck"))

                                cls_grads = optimiser.compute_gradients(cls_gram_loss)
                                self.cls_gram_losses[gram].append(cls_gram_loss)
                                self.cls_gram_probs[gram].append(cls_gram_probs)
                                self.cls_gram_results[gram].append(cls_gram_result)
                                self.cls_gram_ys[gram].append(cls_gram_y)
                                self.cls_gram_weights[gram].append(cls_gram_weights)
                                self.cls_gram_dev_grads[gram].append(cls_grads)

                                for metric_func in self.metric_funcs:
                                    self._create_cls_gram_metric(gram, metric_func[1], metric_func[0], cls_gram_y, cls_gram_result)

                    with tf.variable_scope('Classification', reuse=tf.AUTO_REUSE) as scope:
                        cls_y = tf.placeholder(dtype=tf.int32, shape=(None,), name='Y')
                        cls_top_k = tf.placeholder(dtype=tf.int32, name='TopK')
                        cls_weights = tf.placeholder(dtype=tf.float32, shape=(None,), name='Weight')

                        settings = self._cls_settings
                        x_emd_init = tf.random_normal((self._chars_count, settings['char_vector_size']))
                        x_emb = tf.get_variable("Embeddings", initializer=x_emd_init)
                        rnn_input = tf.nn.embedding_lookup(x_emb, x)

                        if self._for_usage:
                            cls_keep_drop = tf.constant(1, dtype=tf.float32, name='KeepDrop')
                        else:
                            cls_keep_drop = tf.placeholder(dtype=tf.float32, name='KeepDrop')
                        self.cls_keep_drops.append(cls_keep_drop)

                        init_state = [self.cls_gram_probs[gram][-1] for gram in self.cls_gram_probs]
                        init_state = tf.concat(init_state, 1)

                        with tf.variable_scope("InitRnnState", reuse=tf.AUTO_REUSE) as scope:
                            rez_size = settings['rnn_state_size']
                            w_softmax = tf.get_variable("W", (init_state.shape[1], rez_size))
                            b_softmax = tf.get_variable("b", [rez_size])
                            init_state = tf.matmul(init_state, w_softmax) + b_softmax

                        rnn_logits = tfu.build_rnn(rnn_input, cls_keep_drop, seq_len, settings, init_state, init_state, self._for_usage)

                        cls_logits = tfu.rnn_top('RnnTop', rnn_logits, settings, self._classes_count)

                        cls_gram_errors = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_logits,
                                                                                         labels=cls_y)
                        cls_gram_errors = cls_gram_errors * cls_weights
                        if not self._for_usage:
                            self.checks.append(tf.check_numerics(cls_gram_errors, "ErrorNullCheck"))

                        cls_probs = tf.nn.softmax(cls_logits)
                        cls_result = tf.math.top_k(cls_probs, cls_top_k, name="Results")
                        cls_loss = tf.reduce_sum(cls_gram_errors)

                        if not self._for_usage:
                            self.checks.append(tf.check_numerics(cls_gram_errors, "LossNullCheck"))

                        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Classification')
                        cls_grads = optimiser.compute_gradients(cls_loss, var_list=vars)

                        self.cls_losses.append(cls_loss)
                        self.cls_probs.append(cls_probs)
                        self.cls_results.append(cls_result)
                        self.cls_ys.append(cls_y)
                        self.cls_dev_grads.append(cls_grads)
                        self.cls_top_ks.append(cls_top_k)
                        self.cls_weights.append(cls_weights)

                        for metric_func in self.metric_funcs:
                            metr_epoch_loss, metr_update, metr_reset = tfu.create_reset_metric(
                                metric_func[1],
                                metric_func[0],
                                labels=cls_y,
                                predictions=tf.math.argmax(cls_probs, axis=1)
                            )
                            self.cls_metrics_reset.append(metr_reset)
                            self.cls_metrics_update.append(metr_update)
                            self.cls_devices_metrics[metric_func[0]].append(metr_epoch_loss)



            self.cls_gram_grads = {
                gram:  tfu.average_gradients(self.cls_gram_dev_grads[gram])
                for gram in self.cls_gram_dev_grads
            }
            self.cls_gram_optimize = {
                gram: optimiser.apply_gradients(self.cls_gram_grads[gram], name='GramOptimize')
                for gram in self.cls_gram_grads
            }

            self.cls_gram_global_loss = {
                gram: tf.reduce_sum(self.cls_gram_losses[gram], name='GramGlobalLoss')
                for gram in self.cls_gram_losses
            }
            self.cls_gram_metrics = {
                gram: {
                    metr : tf.reduce_mean(self.cls_gram_devices_metrics[gram][metr], name=f"Cls_Gram_{metr}_{gram}")
                    for metr in self.cls_gram_devices_metrics[gram]
                }
                for gram in self.cls_gram_devices_metrics
            }


            self.cls_grads = tfu.average_gradients(self.cls_dev_grads)
            self.cls_optimize = optimiser.apply_gradients(self.cls_grads, name='ClsOptimize')
            self.cls_global_loss = tf.reduce_sum(self.cls_losses, name='ClsGlobalLoss')
            self.cls_metrics = {
                metr: tf.reduce_mean(self.cls_devices_metrics[metr], name=f"Cls_{metr}")
                for metr in self.cls_devices_metrics
            }

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self._checkpoints_keep)



    def _create_cls_gram_metric(self, gram,
                                metric_func,
                                metric_name,
                                labels,
                                predictions):

        epoch_loss, epoch_loss_update, epoch_loss_reset = tfu.create_reset_metric(
            metric_func,
            metric_name,
            labels=labels,
            predictions=predictions
        )

        self.cls_gram_metrics_reset[gram].append(epoch_loss_reset)
        self.cls_gram_metrics_update[gram].append(epoch_loss_update)
        self.cls_gram_devices_metrics[gram][metric_name].append(epoch_loss)







    def train(self, with_restore=True):
        epoch = 1
        config = tf.ConfigProto(allow_soft_placement=True)
        #shutil.rmtree(self._save_path)
        if not os.path.isdir(self._save_path):
            os.mkdir(self._save_path)

        with tf.Session(config = config, graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            if with_restore:
                latest_checkpiont = tf.train.latest_checkpoint(self._save_path)
                if latest_checkpiont:
                    self.saver.restore(sess, latest_checkpiont)

            tqdm.write(f"Start training")
            for gram in self._grammemes:

                if gram in self._miss_steps:
                    print(f"Missing step '{gram}' step")
                    continue

                settings = self._cls_gram_settings[gram]
                tqdm.write(f"Loading dataset for '{gram}'")

                trains = list(tfu.load_dataset(self._dataset_path, self._devices_count, 'train', settings['train_batch_size'], settings['use_weights'], gram))
                valids = list(tfu.load_dataset(self._dataset_path, self._devices_count, 'valid', settings['test_batch_size'], settings['use_weights'], gram))

                tqdm.write(f"Start training classification '{gram}'")

                best_epoch, best_model_acc = self._train_classification(sess,
                                                                        settings,
                                                                        epoch,
                                                                        trains,
                                                                        valids,
                                                                        self.cls_gram_ys[gram],
                                                                        self.cls_gram_weights[gram],
                                                                        self.cls_gram_keep_drops[gram],
                                                                        self.cls_gram_metrics[gram],
                                                                        self.cls_gram_metrics_reset[gram],
                                                                        self.cls_gram_metrics_update[gram],
                                                                        self.cls_gram_optimize[gram],
                                                                        self.cls_learn_rate,
                                                                        f"CLASSIFICATION {gram}"
                                                                        )

                tqdm.write(f"Best model for '{gram}' at epoch {best_epoch}. Acc: {best_model_acc}")


            settings = self._cls_settings
            trains = list(
                tfu.load_dataset(self._dataset_path, self._devices_count, 'train', settings['train_batch_size'], settings['use_weights']))
            valids = list(
                tfu.load_dataset(self._dataset_path, self._devices_count, 'valid', settings['test_batch_size'], settings['use_weights']))

            tqdm.write(f"Start training classification")


            ad_feed_dict = {
                op : 1
                for key in self.cls_gram_keep_drops
                for op in self.cls_gram_keep_drops[key]
            }
            best_epoch, best_model_acc = self._train_classification(sess,
                                                                    settings,
                                                                    epoch,
                                                                    trains,
                                                                    valids,
                                                                    self.cls_ys,
                                                                    self.cls_weights,
                                                                    self.cls_keep_drops,
                                                                    self.cls_metrics,
                                                                    self.cls_metrics_reset,
                                                                    self.cls_metrics_update,
                                                                    self.cls_optimize,
                                                                    self.cls_learn_rate,
                                                                    "CLASSIFICATION GLOBAL",
                                                                    ad_feed_dict
                                                                    )

            tqdm.write(f"Best model for global classification at epoch {best_epoch}. Acc: {best_model_acc}")
            tqdm.write("Training classification finished")
            tqdm.write(self._filler)
            tqdm.write(self._filler)
            tqdm.write(self._filler)
            tqdm.write(self._filler)
            tqdm.write("Start classification testing")

            tests = list(
                tfu.load_dataset(self._dataset_path, self._devices_count, 'test', settings['test_batch_size'], settings['use_weights']))
            test_acc = self._test_step(sess,
                                       tests,
                                       "Testing",
                                       epoch,
                                       self.cls_ys,
                                       self.cls_weights,
                                       self.cls_keep_drops,
                                       self.cls_metrics,
                                       self.cls_metrics_reset,
                                       self.cls_metrics_update,
                                       ad_feed_dict
                                       )
            tqdm.write(f"Test acc for global classification: {test_acc}")


            for gram in self._grammemes:
                settings = self._cls_gram_settings[gram]
                tests = list(tfu.load_dataset(self._dataset_path, self._devices_count, 'test', settings['test_batch_size'], settings['use_weights'], gram))

                test_acc = self._test_step(sess,
                                           tests,
                                           "Testing",
                                           epoch,
                                           self.cls_gram_ys[gram],
                                           self.cls_gram_weights[gram],
                                           self.cls_gram_keep_drops[gram],
                                           self.cls_gram_metrics[gram],
                                           self.cls_gram_metrics_reset[gram],
                                           self.cls_gram_metrics_update[gram]
                                          )
                tqdm.write(f"Test acc for '{gram}': {test_acc}")


    def _train_classification(self,
                              sess,
                              settings,
                              epoch,
                              trains,
                              valids,
                              ys,
                              weights,
                              keep_drops,
                              metrics_val,
                              metrics_reset,
                              metrics_update,
                              optimize,
                              learn_rate_op,
                              operation_name,
                              add_feed_dict=None
                              ):
        best_model_acc = -1
        best_epoch = -1
        learn_rate_val = settings['learn_rate']
        acc_delta = 100
        return_step = 0
        while settings['stop_training_acc_delta'] < acc_delta:
            tqdm.write(self._filler)
            tqdm.write(self._filler)
            tqdm.write(operation_name)

            sess.run(metrics_reset)
            for item in tqdm(trains, desc=f"Train, epoch {epoch}"):
                launch = [
                    optimize
                ]
                launch.extend(metrics_update)
                feed_dic = self._create_feed_dict(item, ys, keep_drops, settings['keep_dropout'], weights)
                feed_dic[learn_rate_op] = learn_rate_val
                if add_feed_dict is not None:
                    feed_dic.update(add_feed_dict)
                sess.run(launch, feed_dic)

            train_acc = self._write_metrics_report(sess, metrics_val, "Train")

            valid_acc = self._test_step(sess,
                                        valids,
                                        "Validation",
                                        epoch,
                                        ys,
                                        weights,
                                        keep_drops,
                                        metrics_val,
                                        metrics_reset,
                                        metrics_update,
                                        add_feed_dict
                                        )

            tqdm.write(f"Epoch {epoch} Train accuracy: {train_acc} Validation accuracy: {valid_acc}")
            need_decay = False
            if valid_acc > best_model_acc:
                if (valid_acc - best_model_acc) < settings['stop_training_acc_delta']:
                    tqdm.write(f"Acc delta is less then min value")
                    need_decay = True
                else:
                    return_step = 0

                best_model_acc = valid_acc
                best_epoch = epoch
                self.saver.save(sess, self._save_path, epoch)
                epoch += 1
            else:
                tqdm.write("Best epoch is better then current")
                if return_step == settings['return_step']:
                    tqdm.write(f"Restoring best epoch {best_epoch}")
                    self.saver.restore(sess, os.path.join(self._save_path, f"-{best_epoch}"))
                need_decay = True

            if need_decay:
                if return_step == settings['return_step']:
                    learn_rate_val = learn_rate_val * settings['learn_rate_decay_step']
                    if learn_rate_val < settings['min_learn_rate']:
                        tqdm.write(f"Learning rate {learn_rate_val} is less then min learning rate")
                        break
                    tqdm.write(f"Learning rate decayed. New value: {learn_rate_val}")
                    return_step = 0
                else:
                    tqdm.write(f"Return step increased")
                    return_step += 1

        return best_epoch, best_model_acc

    def _test_step(self, sess,
                         items,
                         op_name,
                         epoch,
                         ys,
                         weights,
                         keep_drops_op,
                         metrics_val,
                         metrics_reset,
                         metrics_update,
                         add_feed_dict=None):

        sess.run(metrics_reset)
        descr = op_name if epoch is None else f"{op_name}, epoch {epoch}"
        for item in tqdm(items, desc=descr):
            launch = []
            launch.extend(metrics_update)
            feed_dic = self._create_feed_dict(item, ys, keep_drops_op, 1, weights)
            if add_feed_dict is not None:
                feed_dic.update(add_feed_dict)
            sess.run(launch, feed_dic)

        return self._write_metrics_report(sess, metrics_val, op_name)

    def _write_metrics_report(self, sess, metrics, step_name):
        tqdm.write('')
        tqdm.write(f"{step_name}")

        launch_results = sess.run(metrics)

        i=0
        result = []
        for metr in metrics:
            result.append('{:>8}'.format(metr))
            result.append("=")
            result.append("{0:.7f}".format(launch_results[metr]))
            result.append(" ")
            i+=1
        result = "".join(result)
        tqdm.write(result)
        return launch_results["Accuracy"]




    def _create_feed_dict(self, item, ys, keep_drops, dropout_vals, weights):
        feed_dic = {}
        for dev_num, batch in enumerate(item):
            feed_dic[self.xs[dev_num]] = batch['x']
            feed_dic[self.seq_lens[dev_num]] = batch['seq_len']
            feed_dic[keep_drops[dev_num]] = dropout_vals
            feed_dic[ys[dev_num]] = batch['ys']
            feed_dic[weights[dev_num]] = batch['weight']

        return feed_dic


    def release(self):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # Loading checkpoint
            latest_checkpiont = tf.train.latest_checkpoint(self._save_path)
            if latest_checkpiont:
                self.saver.restore(sess, latest_checkpiont)

            if os.path.isdir(self._export_path):
                shutil.rmtree(self._export_path)


            output_dic = {}
            gram_op_dic = {}
            for gram in self.cls_gram_results:
                res = self.cls_gram_results[gram][0]
                prob = self.cls_gram_probs[gram][0]
                output_dic[f'res_{gram}'] = res
                output_dic[f'prob_{gram}'] = prob
                gram_op_dic[gram] = {
                    'res': res.op.name,
                    'prob': prob.op.name
                }

            output_dic['res_values'] = self.cls_results[0][0]
            output_dic['res_indexes'] = self.cls_results[0][1]
            # Saving model
            tf.saved_model.simple_save(sess,
                                       self._export_path,
                                       inputs={
                                           'x': self.xs[0],
                                           'seq_len': self.seq_lens[0],
                                           'top_k': self.cls_top_ks[0]
                                       },
                                       outputs=output_dic)

            # Freezing graph
            input_graph = 'graph.pbtxt'
            tf.train.write_graph(sess.graph.as_graph_def(), self._export_path, input_graph, as_text=True)
            input_graph = os.path.join(self._export_path, input_graph)
            frozen_path = os.path.join(self._export_path, 'frozen_model.pb')
            output_ops = [output_dic[key].op.name for key in output_dic]
            output_ops = ",".join(output_ops)
            freeze_graph.freeze_graph(input_graph,
                                      "",
                                      False,
                                      latest_checkpiont,
                                      output_ops,
                                      "",
                                      "",
                                      frozen_path,
                                      True,
                                      "",
                                      input_saved_model_dir=self._export_path)


            return frozen_path, \
                   self._classes_dic, \
                   gram_op_dic, \
                   self.cls_results[0][0].op.name

