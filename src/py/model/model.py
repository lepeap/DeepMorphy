import os
import pickle
import shutil
import tensorflow as tf
from graph.gram_cls import GramCls
from graph.main_cls import MainCls
from graph.lemm import Lemm
from graph.base import TfContext
from utils import MyDefaultDict, config
from tensorflow.python.tools import freeze_graph


class RNN:
    def __init__(self, for_usage):
        self.config = config()
        self.filler = self.config['filler']
        self.checkpoints_keep = 200000
        self.for_usage = for_usage
        self.default_config = self.config['graph_part_configs']['default']
        self.key_configs = MyDefaultDict(
            lambda key: self.default_config,
            {
                key: MyDefaultDict(lambda prop_key: self.default_config[prop_key], self.config['graph_part_configs'][key])
                for key in self.config['graph_part_configs']
                if key != 'default'
            }
        )
        self.export_path = self.config['export_path']
        self.save_path = self.config['save_path']
        self.publish_path = self.config['publish_net_path']
        self.model_key = self.config['model_key']
        self.miss_steps = self.config['miss_steps'] if 'miss_steps' in self.config else []
        self.start_char = self.config['start_token']
        self.end_char = self.config['end_token']
        self.gram_keys = [
            key
            for key in sorted(self.config['grammemes_types'], key=lambda x: self.config['grammemes_types'][x]['index'])
        ]
        self.main_class_k = self.config['main_class_k']
        self.train_steps = self.config['train_steps']

        with open(os.path.join(self.config['dataset_path'], f"classification_classes.pkl"), 'rb') as f:
            self.config['main_classes'] = pickle.load(f)
            self.config['main_classes_count'] = len(self.config['main_classes'])

        if for_usage:
            self.devices = ['/cpu:0']
        else:
            self.devices = self.config['train_devices']

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

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
            self.batch_size = tf.placeholder(tf.int32, [], name="BatchSize") if self.for_usage else None
            self.optimiser = tf.train.AdamOptimizer(self.learn_rate)
            self.gram_graph_parts = {
                gram: GramCls(gram, self.for_usage, self.config, self.key_configs[gram], self.optimiser)
                for gram in self.gram_keys
            }
            self.lem_graph_part = Lemm(self.for_usage, self.config, self.key_configs["lemm"], self.optimiser)
            self.main_graph_part = MainCls(self.for_usage, self.config, self.key_configs["main"], self.optimiser)

            for device_index, device_name in enumerate(self.devices):
                with tf.device(device_name):
                    if self.for_usage:
                        x_ind_pl = tf.placeholder(dtype=tf.int32, shape=(None, None), name='XIndexes')
                        x_val_pl = tf.placeholder(dtype=tf.int32, shape=(None,), name='XValues')
                        x_shape_pl = tf.placeholder(dtype=tf.int32, shape=(2,), name='XShape')
                        x_ind =   tf.dtypes.cast(x_ind_pl, dtype=tf.int64)
                        x_val =   tf.dtypes.cast(x_val_pl, dtype=tf.int64)
                        x_shape = tf.dtypes.cast(x_shape_pl, dtype=tf.int64)

                        x_sparse = tf.sparse.SparseTensor(x_ind, x_val, x_shape)
                        x = tf.sparse.to_dense(x_sparse, default_value=self.end_char)
                        self.x_inds.append(x_ind_pl)
                        self.x_vals.append(x_val_pl)
                        self.x_shape.append(x_shape_pl)
                    else:
                        x = tf.placeholder(dtype=tf.int32, shape=(None, None), name='X')
                        self.xs.append(x)

                    seq_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='SeqLen')
                    self.seq_lens.append(seq_len)

                    for gram in self.gram_keys:
                        self.gram_graph_parts[gram].build_graph_for_device(x, seq_len)

                    gram_probs = [self.gram_graph_parts[gram].probs[-1] for gram in self.gram_keys]
                    gram_keep_drops = [self.gram_graph_parts[gram].keep_drops[-1] for gram in self.gram_keys]
                    self.main_graph_part.build_graph_for_device(x, seq_len, gram_probs, gram_keep_drops)
                    self.prints.append(tf.print("main_result", self.main_graph_part.results[0].indices))
                    if self.for_usage and not self.lem_graph_part.use_cls_placeholder:
                        x = tf.contrib.seq2seq.tile_batch(x, multiplier=self.main_class_k)
                        seq_len = tf.contrib.seq2seq.tile_batch(seq_len, multiplier=self.main_class_k)
                        cls = tf.reshape(self.main_graph_part.results[0].indices, (-1,))
                        batch_size = self.batch_size * self.main_class_k
                        self.lem_graph_part.build_graph_for_device(x,
                                                                   seq_len,
                                                                   batch_size,
                                                                   cls)
                        self.lem_result = tf.reshape(self.lem_graph_part.results[0],
                                                     (self.batch_size, self.main_class_k, -1))

                    else:
                        self.lem_graph_part.build_graph_for_device(x,
                                                                   seq_len,
                                                                   self.batch_size)

            for gram in self.gram_keys:
                self.gram_graph_parts[gram].build_graph_end()
            self.main_graph_part.build_graph_end()
            self.lem_graph_part.build_graph_end()
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.checkpoints_keep)

    def train(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        with tf.Session(config = config, graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            tc = TfContext(sess, self.saver, self.learn_rate)
            self.__restore__(tc)

            for gram in self.gram_keys:
                if gram in self.train_steps:
                    self.gram_graph_parts[gram].train(tc)

            if self.main_graph_part.key in self.train_steps:
                self.main_graph_part.train(tc)

            if self.lem_graph_part.key in self.train_steps:
                self.lem_graph_part.train(tc)


    def __restore__(self, tc):
        latest_checkpiont = tf.train.latest_checkpoint(self.save_path)
        for gram in self.gram_keys:
            if gram not in self.config['ignore_restore']:
                self.gram_graph_parts[gram].restore(tc, latest_checkpiont)
        if self.main_graph_part.key not in self.config['ignore_restore']:
            self.main_graph_part.restore(tc, latest_checkpiont)
        if self.lem_graph_part.key not in self.config['ignore_restore']:
            self.lem_graph_part.restore(tc, latest_checkpiont)







    def release(self):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # Loading checkpoint
            latest_checkpoint = tf.train.latest_checkpoint(self.save_path)
            if latest_checkpoint:
                self.saver.restore(sess, latest_checkpoint)

            if os.path.isdir(self.export_path):
                shutil.rmtree(self.export_path)


            output_dic = {}
            gram_op_dic = {}
            for gram in self.gram_keys:
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
                                       self.export_path,
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
            tf.train.write_graph(sess.graph.as_graph_def(), self.export_path, input_graph, as_text=True)
            input_graph = os.path.join(self.export_path, input_graph)
            frozen_path = os.path.join(self.export_path, 'frozen_model.pb')
            output_ops = [output_dic[key].op.name for key in output_dic]
            output_ops = ",".join(output_ops)
            freeze_graph.freeze_graph(input_graph,
                                      "",
                                      False,
                                      latest_checkpoint,
                                      output_ops,
                                      "",
                                      "",
                                      frozen_path,
                                      True,
                                      "",
                                      input_saved_model_dir=self.export_path)

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
                   self.config['main_classes'], \
                   gram_op_dic , \
                   op_dic

