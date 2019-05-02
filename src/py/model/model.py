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
        self._lemmatizer_settings = self.config['lemmatizer']
        self._miss_steps = self.config['miss_steps'] if 'miss_steps' in self.config else []
        self._start_char_index = self.config['start_token']
        self._end_char_index = self.config['end_token']
        self._filler = "############################################################################"

        with open(os.path.join(self._dataset_path, f"classification_classes.pkl"), 'rb') as f:
            self._classes_dic = pickle.load(f)
            self._classes_count = len(self._classes_dic)

        if for_usage:
            self._devices = ['/cpu:0']
        else:
            self._devices = self.config['train_devices']

        self._devices_count = len(self._devices)
        if not os.path.exists(self._save_path):
            os.mkdir(self._save_path)

        self._build_graph()


    def _build_graph(self):
        self.graph = tf.Graph()

        self.prints = []

        self.checks = []
        self.xs = []
        self.seq_lens = []
        self.metric_funcs = [
            ('Accuracy', tf.metrics.accuracy)
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
        self.lem_ys_seq_len = []
        self.lem_cls = []
        self.lem_grads = []
        self.lem_losses = []
        self.lem_accs = []
        self.lem_results = []
        self.lem_metrics_reset = []
        self.lem_metrics_update = []
        self.lem_devices_metrics = {metr[0]: [] for metr in self.metric_funcs}

        with self.graph.as_default(), tf.device('/cpu:0'):
            self.is_training = tf.placeholder(tf.bool, name="IsTraining")
            self.learn_rate = tf.placeholder(tf.float32, name="LearningRate")
            optimiser = tf.train.AdamOptimizer(self.learn_rate)
            for deviceIndex, deviceName in enumerate(self._devices):
                with tf.device(deviceName):
                    x = tf.placeholder(dtype=tf.int32, shape=(None, None), name='X')
                    seq_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='SeqLen')
                    self.seq_lens.append(seq_len)
                    self.xs.append(x)

                    with tf.variable_scope('Lemmatizer', reuse=tf.AUTO_REUSE) as scope:
                        settings = self._lemmatizer_settings
                        lemma_cls = tf.placeholder(dtype=tf.int32, shape=(None,), name='XClass')
                        lemma_y = tf.placeholder(dtype=tf.int32, shape=(None, None), name='Y')
                        lemma_y_seq_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='YSeqLen')



                        lemma_cls_init = tf.random_normal((self._classes_count, settings['encoder']['rnn_state_size']))
                        lemma_cls_emb = tf.get_variable("EmbeddingsLemmaCls", initializer=lemma_cls_init)
                        init_state = tf.nn.embedding_lookup(lemma_cls_emb, lemma_cls)


                        x_emd_init = tf.random_normal((self._chars_count+2, settings['char_vector_size']))
                        x_emb = tf.get_variable("EmbeddingsX", initializer=x_emd_init)
                        encoder_input = tf.nn.embedding_lookup(x_emb, x)

                        y_emd_init = tf.random_normal((self._chars_count+2, settings['char_vector_size']))
                        y_emb = tf.get_variable("EmbeddingsY", initializer=y_emd_init)
                        lemma_output = tf.nn.embedding_lookup(y_emb, lemma_y)

                        if self._for_usage:
                            keep_drop = tf.constant(1, dtype=tf.float32, name='KeepDrop')
                        else:
                            keep_drop = tf.placeholder(dtype=tf.float32, name='KeepDrop')


                        with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE) as scope:
                            encoder_output = tfu.build_rnn(
                                                       encoder_input,
                                                       keep_drop,
                                                       seq_len,
                                                       settings['encoder'],
                                                       init_state,
                                                       init_state,
                                                       self._for_usage
                            )

                        with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE) as scope:

                            if self._for_usage:
                                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(encoder_output,
                                                                                  start_token=self._start_char_index,
                                                                                  end_token=self._end_char_index)
                            else:
                                helper = tf.contrib.seq2seq.TrainingHelper(lemma_output,
                                                                           [settings['max_length'] for _ in range(settings['batch_size'])])

                            decoder_output = tfu.decoder(
                                helper,
                                encoder_output,
                                seq_len,
                                settings,
                                self._chars_count+2,
                                settings['batch_size'],
                                self._for_usage
                            )
                            #self.prints.append(tf.print("decoder output", tf.shape(decoder_output.rnn_output)))

                        #self.prints.append(tf.print("y output", tf.shape(lemma_y)))
                        masks = tf.sequence_mask(
                            lengths=lemma_y_seq_len,
                            dtype=tf.float32,
                            maxlen=settings['max_length']
                        )
                        #self.prints.append(tf.print("weight output", tf.shape(masks)))

                        loss = tf.contrib.seq2seq.sequence_loss(decoder_output.rnn_output, lemma_y, masks)
                        #self.prints.append(tf.print("loss", loss))
                        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Lemmatizer')
                        cls_grads = optimiser.compute_gradients(loss, var_list=vars)

                        for metric_func in self.metric_funcs:
                            seq_mask = tf.cast(tf.reshape(masks, (-1,)), tf.int32)
                            nonzero_indices = tf.where(tf.not_equal(seq_mask, 0))

                            labels = tf.reshape(lemma_y, (-1,))
                            labels = tf.gather(labels, nonzero_indices)

                            predictions = tf.reshape(decoder_output.sample_id, (-1,))
                            predictions = tf.gather(predictions, nonzero_indices)

                            #self.prints.append(tf.print("non zero indexes output", nonzero_indices))
                            #self.prints.append(tf.print("labels output", labels))
                            #self.prints.append(tf.print("predictions output", predictions))


                            metr_epoch_loss, metr_update, metr_reset = tfu.create_reset_metric(
                                metric_func[1],
                                metric_func[0],
                                labels=labels,
                                predictions=predictions
                            )
                            self.lem_metrics_reset.append(metr_reset)
                            self.lem_metrics_update.append(metr_update)
                            self.lem_devices_metrics[metric_func[0]].append(metr_epoch_loss)

                        self.lem_ys.append(lemma_y)
                        self.lem_ys_seq_len.append(lemma_y_seq_len)
                        self.lem_cls.append(lemma_cls)
                        self.lem_grads.append(cls_grads)
                        self.lem_losses.append(loss)


                    with tf.variable_scope('GramClassification', reuse=tf.AUTO_REUSE) as scope:
                        for gram in self._grammemes:
                            with tf.variable_scope(gram, reuse=tf.AUTO_REUSE) as scope:
                                if self._for_usage:
                                    cls_gram_keep_drop = tf.constant(1, dtype=tf.float32, name='KeepDrop')
                                else:
                                    cls_gram_keep_drop = tf.placeholder(dtype=tf.float32, name='KeepDrop')

                                self.cls_gram_keep_drops[gram].append(cls_gram_keep_drop)

                                cls_gram_y = tf.placeholder(dtype=tf.int32, shape=(None, len(self._grammemes[gram]['classes'])), name='Y')
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

                                float_gram_y = tf.cast(cls_gram_y, tf.float32)
                                cls_gram_errors = tf.nn.sigmoid_cross_entropy_with_logits(logits=cls_gram_logits, labels=float_gram_y)
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
                                    labels = tf.math.argmax(cls_gram_y, axis=1)
                                    predictions = cls_gram_result
                                    self._create_cls_gram_metric(gram,
                                                                 metric_func[1],
                                                                 metric_func[0],
                                                                 labels,
                                                                 predictions
                                                                 )

                    with tf.variable_scope('Classification', reuse=tf.AUTO_REUSE) as scope:
                        cls_y = tf.placeholder(dtype=tf.int32, shape=(None, self._classes_count), name='Y')
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

                        float_y = tf.cast(cls_y, tf.float32)
                        cls_gram_errors = tf.nn.sigmoid_cross_entropy_with_logits(logits=cls_logits,
                                                                                  labels=float_y)
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
                            labels = tf.math.argmax(cls_y, axis=1)
                            predictions = tf.math.argmax(cls_probs, axis=1)
                            metr_epoch_loss, metr_update, metr_reset = tfu.create_reset_metric(
                                metric_func[1],
                                metric_func[0],
                                labels=labels,
                                predictions=predictions
                            )
                            self.cls_metrics_reset.append(metr_reset)
                            self.cls_metrics_update.append(metr_update)
                            self.cls_devices_metrics[metric_func[0]].append(metr_epoch_loss)



            self.lem_grads = tfu.average_gradients(self.lem_grads)
            self.lem_optimize = optimiser.apply_gradients(self.lem_grads, name='LemOptimize')
            self.lem_global_loss = tf.reduce_sum(self.lem_losses, name='LemGlobalLoss')
            self.lem_metrics = {
                metr: tf.reduce_mean(self.lem_devices_metrics[metr], name=f"Lem_{metr}")
                for metr in self.lem_devices_metrics
            }


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

            self._train_lemmatization(sess, 0)
            tqdm.write(f"Start training")

            for gram in self._grammemes:

                if gram in self._miss_steps:
                    print(f"Missing step '{gram}' step")
                    continue

                settings = self._cls_gram_settings[gram]
                tqdm.write(f"Loading dataset for '{gram}'")

                trains = list(tfu.load_cls_dataset(self._dataset_path,
                                                   self._devices_count,
                                                   'train',
                                                   settings['train_batch_size'],
                                                   settings['use_weights'], gram))

                valids = list(tfu.load_cls_dataset(self._dataset_path,
                                                   self._devices_count,
                                                   'valid',
                                                   settings['test_batch_size'],
                                                   settings['use_weights'], gram))

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
                                                                        self.learn_rate,
                                                                        f"CLASSIFICATION {gram}"
                                                                        )

                tqdm.write(f"Best model for '{gram}' at epoch {best_epoch}. Acc: {best_model_acc}")


            settings = self._cls_settings
            trains = list(
                tfu.load_cls_dataset(self._dataset_path, self._devices_count, 'train', settings['train_batch_size'], settings['use_weights']))
            valids = list(
                tfu.load_cls_dataset(self._dataset_path, self._devices_count, 'valid', settings['test_batch_size'], settings['use_weights']))

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
                                                                    self.learn_rate,
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
            settings = self._lemmatizer_settings
            lem_tests = list( tfu.load_lemma_dataset(self._dataset_path,
                                                     self._devices_count,
                                                     'train',
                                                     settings['batch_size']))
            test_acc = self._test_lem_steo(sess, lem_tests, "Testing", epoch)
            tqdm.write(f"Test acc for lemmatization: {test_acc}")


            settings = self._cls_settings
            tests = list(tfu.load_cls_dataset(self._dataset_path,
                                              self._devices_count,
                                              'test',
                                              settings['test_batch_size'],
                                              settings['use_weights']))
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
                tests = list(tfu.load_cls_dataset(self._dataset_path, self._devices_count, 'test', settings['test_batch_size'], settings['use_weights'], gram))

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


    def _train_lemmatization(self, sess, epoch):
        settings = self._lemmatizer_settings
        trains = list(
            tfu.load_lemma_dataset(self._dataset_path, self._devices_count, 'train', settings['batch_size']))
        valids = list(
            tfu.load_lemma_dataset(self._dataset_path, self._devices_count, 'valid', settings['batch_size']))

        best_model_acc = -1
        best_epoch = -1
        learn_rate_val = settings['learn_rate']
        acc_delta = 100
        return_step = 0
        while settings['stop_training_acc_delta'] < acc_delta:
            tqdm.write(self._filler)
            tqdm.write(self._filler)
            tqdm.write("Lemmatization")

            sess.run(self.lem_metrics_reset)
            for item in tqdm(trains, desc=f"Train, epoch {epoch}"):
                launch = []
                launch.extend(self.prints)
                launch.append(self.lem_optimize)
                launch.extend(self.lem_metrics_update)
                feed_dic = {}
                for i in range(len(item)):
                    feed_dic[self.xs[i]] = item[i]['x']
                    feed_dic[self.seq_lens[i]] = item[i]['x_seq_len']
                    feed_dic[self.lem_cls[i]] = item[i]['x_cls']
                    feed_dic[self.lem_ys[i]] = item[i]['y']
                    feed_dic[self.lem_ys_seq_len[i]] = item[i]['y_seq_len']

                feed_dic[self.learn_rate] = learn_rate_val
                sess.run(launch, feed_dic)

            train_acc = self._write_metrics_report(sess, self.lem_metrics, "Train")
            valid_acc = self._test_lem_steo(sess, valids, "Validation", epoch)

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

    def _test_lem_steo(self, sess, items, op_name, epoch):
        sess.run(self.lem_metrics_reset)
        descr = op_name if epoch is None else f"{op_name}, epoch {epoch}"
        for item in tqdm(items, desc=descr):
            launch = []
            launch.extend(self.lem_metrics_update)
            feed_dic = {}
            for i in range(len(item)):
                feed_dic[self.xs[i]] = item[i]['x']
                feed_dic[self.seq_lens[i]] = item[i]['x_seq_len']
                feed_dic[self.lem_cls[i]] = item[i]['x_cls']
                feed_dic[self.lem_ys[i]] = item[i]['y']
                feed_dic[self.lem_ys_seq_len[i]] = item[i]['y_seq_len']

            sess.run(launch, feed_dic)

        return self._write_metrics_report(sess, self.lem_metrics, op_name)

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
            feed_dic[self.seq_lens[dev_num]] = batch['x_seq_len']
            feed_dic[keep_drops[dev_num]] = dropout_vals
            feed_dic[ys[dev_num]] = batch['y']
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

