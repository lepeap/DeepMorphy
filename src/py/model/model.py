import os
import numpy as np
import yaml
import pickle
import shutil
import tensorflow as tf
import tf_utils as tfu
from tqdm import tqdm
from graph.gram_cls import GramCls
from graph.main_cls import MainCls
from graph.lemm import Lemm
from graph.base import TrainContext
from utils import MyDefaultDict
from tensorflow.python.tools import freeze_graph


class RNN:
    def __init__(self, for_usage):
        with open('config.yml', 'r') as f:
            self._config = yaml.load(f)
        self._filler = self._config['filler']
        self._checkpoints_keep = 200000
        self._for_usage = for_usage
        self._default_config = self._config['graph_part_configs']['default']
        self._key_configs = MyDefaultDict(
            lambda key: self._default_config,
            {
                key: MyDefaultDict(lambda prop_key: self._default_config[prop_key], self._config['graph_part_configs'][key])
                for key in self._config['graph_part_configs']
                if key != 'default'
            }
        )
        self._export_path = self._config['export_path']
        self._save_path = self._config['save_path']
        self._publish_path = self._config['publish_net_path']
        self._model_key = self._config['model_key']
        self._miss_steps = self._config['miss_steps'] if 'miss_steps' in self._config else []
        self._start_char = self._config['start_token']
        self._end_char = self._config['end_token']
        self._gram_keys = [
            key
            for key in sorted(self._config['grammemes_types'], key=lambda x: self._config['grammemes_types'][x]['index'])
        ]
        self._main_class_k = self._config['main_class_k']
        self._train_steps = self._config['train_steps']

        with open(os.path.join(self._config['dataset_path'], f"classification_classes.pkl"), 'rb') as f:
            self._config['main_classes'] = pickle.load(f)
            self._config['main_classes_count'] = len(self._config['main_classes'])

        if for_usage:
            self._devices = ['/cpu:0']
        else:
            self._devices = self._config['train_devices']

        if not os.path.exists(self._save_path):
            os.mkdir(self._save_path)

        self.__build_graph__()


    def __build_graph__(self):
        self.graph = tf.Graph()
        self.checks = []
        self.xs = []
        self.seq_lens = []

        self.x_inds = []
        self.x_vals = []
        self.x_shape = []

        self.prints = []

        with self.graph.as_default(), tf.device('/cpu:0'):
            self.is_training = tf.placeholder(tf.bool, name="IsTraining")
            self.learn_rate = tf.placeholder(tf.float32, name="LearningRate")
            self.batch_size = tf.placeholder(tf.int32, [], name="BatchSize") if self._for_usage else None
            self.optimiser = tf.train.AdamOptimizer(self.learn_rate)
            self.gram_graph_parts = {
                gram: GramCls(gram, self._for_usage, self._config, self._key_configs[gram], self.optimiser)
                for gram in self._gram_keys
            }
            self.lem_graph_part = Lemm(self._for_usage, self._config, self._key_configs["lemm"], self.optimiser)
            self.main_graph_part = MainCls(self._for_usage, self._config, self._key_configs["main"], self.optimiser)

            for device_index, device_name in enumerate(self._devices):
                with tf.device(device_name):
                    #if self._for_usage:
                    #    x_ind_pl = tf.placeholder(dtype=tf.int32, shape=(None, None), name='XIndexes')
                    #    x_val_pl = tf.placeholder(dtype=tf.int32, shape=(None,), name='XValues')
                    #    x_shape_pl = tf.placeholder(dtype=tf.int32, shape=(2,), name='XShape')
                    #    x_ind =   tf.dtypes.cast(x_ind_pl, dtype=tf.int64)
                    #    x_val =   tf.dtypes.cast(x_val_pl, dtype=tf.int64)
                    #    x_shape = tf.dtypes.cast(x_shape_pl, dtype=tf.int64)
#
                    #    x_sparse = tf.sparse.SparseTensor(x_ind, x_val, x_shape)
                    #    x = tf.sparse.to_dense(x_sparse, default_value=self._end_char)
                    #    self.x_inds.append(x_ind_pl)
                    #    self.x_vals.append(x_val_pl)
                    #    self.x_shape.append(x_shape_pl)
                    #else:
                    #    x = tf.placeholder(dtype=tf.int32, shape=(None, None), name='X')
                    #    self.xs.append(x)

                    x = tf.placeholder(dtype=tf.int32, shape=(None, None), name='X')
                    self.xs.append(x)

                    seq_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='SeqLen')
                    self.seq_lens.append(seq_len)


                    for gram in self._gram_keys:
                        self.gram_graph_parts[gram].build_graph_for_device(x, seq_len)

                    gram_probs = [self.gram_graph_parts[gram].probs[-1] for gram in self._gram_keys]
                    gram_keep_drops = [self.gram_graph_parts[gram].keep_drops[-1] for gram in self._gram_keys]
                    self.main_graph_part.build_graph_for_device(x, seq_len, gram_probs, gram_keep_drops)
                    #self.prints.append(tf.print("main_result", self.main_graph_part.results[0].indices))
                    if self._for_usage:
                        #self.prints.append(tf.print("xs", x))
                        #self.prints.append(tf.print("seq_len", seq_len))
                        lem_results = []
                        lem_results_lengths = []
                        flat_cls_indexes = tf.reshape(self.main_graph_part.results[0].indices, (-1,))
                        seq_len = seq_len
                        self.main_pl_classes = tf.placeholder(dtype=tf.int32, shape=(None,), name='XClass')
                        #self.prints.append(tf.print("seq_len", seq_len))
                        for i in range(self._main_class_k):
                            indexes = tf.range(i, self.batch_size * self._main_class_k, self._main_class_k)
                            #self.prints.append(tf.print("indexes", indexes))
                            classes = tf.gather(flat_cls_indexes, indexes)
                            #self.prints.append(tf.print("classes", classes))
                            lem_part = Lemm(self._for_usage, self._config, self._key_configs["lemm"], self.optimiser)
                            lem_part.build_graph_for_device(x, seq_len, self.batch_size, self.main_pl_classes)

                            lem_results.append(lem_part.results[0])
                            #lem_results_lengths.append(lem_part.results_lengths[0])
                            #self.prints.extend(lem_part.prints)

                        self.lem_result = tf.stack(lem_results)
                        self.lem_result_length = tf.stack(lem_results_lengths)

                    else:
                        self.lem_graph_part.build_graph_for_device(x,
                                                                   seq_len,
                                                                   self.batch_size
                                                                   )
            if not self._for_usage:
                for gram in self._gram_keys:
                    self.gram_graph_parts[gram].build_graph_end()
                self.main_graph_part.build_graph_end()
                self.lem_graph_part.build_graph_end()
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self._checkpoints_keep)

    def train(self, with_restore=True):
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

            tc = TrainContext(sess, self.saver, self.learn_rate)
            for gram in self._gram_keys:
                if gram in self._train_steps:
                    self.gram_graph_parts[gram].train(tc)

            if self.main_graph_part.key in self._train_steps:
                self.main_graph_part.train(tc)

            if self.lem_graph_part.key in self._train_steps:
                self.lem_graph_part.train(tc)

            tqdm.write(self._filler)
            tqdm.write(self._filler)
            tqdm.write(self._filler)
            tqdm.write(self._filler)
            tqdm.write("Testing")

            for gram in self._gram_keys:
                self.gram_graph_parts[gram].test(tc)

            self.main_graph_part.test(tc)
            self.lem_graph_part.test(tc)

            print()

    def infer(self):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())


            # Loading checkpoint
            latest_checkpiont = tf.train.latest_checkpoint(self._save_path)
            if latest_checkpiont:
                self.saver.restore(sess, latest_checkpiont)
                #self.saver.restore(sess, latest_checkpiont)


            bs = 128
            item = next(tfu.load_lemma_dataset(
                "dataset",
                1,
                "train",
                bs
            ))

            values = np.asarray([
                20,16,14,13,6,15,16,4,16
            ]
            )
            indexes = np.asarray([
                [0, 0],
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [0, 5],
                [0, 6],
                [0, 7],
                [0, 8],
            ])
            seq_len = np.asarray([9])

            launch = [self.lem_result]
            launch.extend(self.prints)

            res = sess.run(
                launch,
                {
                    self.xs[0]: item[0]['x'],
                    self.seq_lens[0]: item[0]['x_seq_len'],
                    self.main_pl_classes: item[0]['x_cls'],
                    self.batch_size: bs
                }
            )
            #self.xs[0]: item['x'],
            #self.x_inds[0]: indexes,
            #self.x_shape[0]: np.asarray([1, 9]),
            #self.seq_lens[0]: item['x_seq_len'],
            #self.batch_size: bs
            print()


    def release(self):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # Loading checkpoint
            latest_checkpiont = tf.train.latest_checkpoint(self._save_path)
            if latest_checkpiont:
                tfu.optimistic_restore(sess, latest_checkpiont)
                #self.saver.restore(sess, latest_checkpiont)

            if os.path.isdir(self._export_path):
                shutil.rmtree(self._export_path)


            output_dic = {}
            gram_op_dic = {}
            for gram in self._gram_keys:
                res = self.gram_graph_parts[gram].results[0]
                prob = self.gram_graph_parts[gram].probs[0]
                output_dic[f'res_{gram}'] = res
                output_dic[f'prob_{gram}'] = prob
                gram_op_dic[gram] = {
                    'res': res.op.name,
                    'prob': prob.op.name
                }

            output_dic['res_values'] = self.main_graph_part.results[0].values
            output_dic['res_indexes'] = self.main_graph_part.results[0].indices
            output_dic['res_lem'] = self.lem_result
            # Saving model
            tf.saved_model.simple_save(sess,
                                       self._export_path,
                                       inputs={
                                           'x_ind': self.x_inds[0],
                                           'x_val': self.x_vals[0],
                                           'x_shape': self.x_shape[0],
                                           'seq_len': self.seq_lens[0],
                                           'batch_size': self.batch_size
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

            op_dic = {
                key: output_dic[key].op.name
                for key in output_dic
            }

            op_dic['x_ind'] = self.x_inds[0].op.name
            op_dic['x_val'] = self.x_vals[0].op.name
            op_dic['x_shape'] = self.x_shape[0].op.name
            op_dic['seq_len'] = self.main_graph_part.seq_lens[0].op.name
            op_dic['batch_size'] = self.batch_size.op.name

            return frozen_path, \
                   self._config['main_classes'], \
                   gram_op_dic , \
                   op_dic

