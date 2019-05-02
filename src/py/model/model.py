import os
import yaml
import pickle
import shutil
import tensorflow as tf
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
        self._gram_keys = [
            key
            for key in sorted(self._config['grammemes_types'], key=lambda x: self._config['grammemes_types'][x]['index'])
        ]

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

        with self.graph.as_default(), tf.device('/cpu:0'):
            self.is_training = tf.placeholder(tf.bool, name="IsTraining")
            self.learn_rate = tf.placeholder(tf.float32, name="LearningRate")
            self.optimiser = tf.train.AdamOptimizer(self.learn_rate)
            self.gram_graph_parts = {
                gram: GramCls(gram, self._for_usage, self._config, self._key_configs[gram], self.optimiser)
                for gram in self._gram_keys
            }
            self.lem_graph_part = Lemm(self._for_usage, self._config, self._key_configs["lemm"], self.optimiser)
            self.main_graph_part = MainCls(self._for_usage, self._config, self._key_configs["main"], self.optimiser)

            for device_index, device_name in enumerate(self._devices):
                with tf.device(device_name):
                    x = tf.placeholder(dtype=tf.int32, shape=(None, None), name='X')
                    seq_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='SeqLen')
                    self.seq_lens.append(seq_len)
                    self.xs.append(x)

                    for gram in self._gram_keys:
                        self.gram_graph_parts[gram].build_graph_for_device(x, seq_len)

                    gram_probs = [self.gram_graph_parts[gram].probs[-1] for gram in self._gram_keys]
                    gram_keep_drops = [self.gram_graph_parts[gram].keep_drops[-1] for gram in self._gram_keys]
                    self.main_graph_part.build_graph_for_device(x, seq_len, gram_probs, gram_keep_drops)
                    self.lem_graph_part.build_graph_for_device(x, seq_len)

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
                self.gram_graph_parts[gram].train(tc)

            self.main_graph_part.train(tc)
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

