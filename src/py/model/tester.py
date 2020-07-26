import os, pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from model import RNN
from utils import CONFIG, decode_word


class Tester:
    def __init__(self):
        self.config = CONFIG
        self.config['graph_part_configs']['lemm']['use_cls_placeholder'] = True
        self.rnn = RNN(True)
        self.chars = {c: index for index, c in enumerate(self.config['chars'])}
        self.batch_size = 4096
        self.show_bad_items = False

    def test(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        results = []
        with tf.Session(config=config, graph=self.rnn.graph) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            self.rnn.restore(sess)

            for gram in self.rnn.gram_keys:
                full_cls_acc, part_cls_acc = self.__test_classification__(sess, gram, self.rnn.gram_graph_parts[gram])
                result = f"{gram}. full_cls_acc: {full_cls_acc}; part_cls_acc: {part_cls_acc}"
                results.append(result)
                tqdm.write(result)
            #
            full_cls_acc, part_cls_acc = self.__test_classification__(sess, 'main', self.rnn.main_graph_part)
            result = f"main. full_cls_acc: {full_cls_acc}; part_cls_acc: {part_cls_acc}"
            results.append(result)
            tqdm.write(result)
            lemm_acc = self.__test_lemmas__(sess)
            result = f"lemma_acc: {lemm_acc}"
            results.append(result)
            tqdm.write(result)

        return "\n".join(results)

    def __get_classification_info__(self, sess, items, graph_part):

        wi = 0
        pbar = tqdm(total=len(items), desc='Getting classification info')
        results = []
        etalon = []

        while wi < len(items):
            bi = 0
            xs = []
            indexes = []
            seq_lens = []
            max_len = 0

            while bi < self.batch_size and wi < len(items):
                word = items[wi]['src']
                etalon.append(items[wi]['y'])
                for c_index, char in enumerate(word):
                    xs.append(self.chars[char] if char in self.chars else self.chars['UNDEFINED'])
                    indexes.append([bi, c_index])
                cur_len = len(word)
                if cur_len > max_len:
                    max_len = cur_len
                seq_lens.append(cur_len)
                bi += 1
                wi += 1
                pbar.update(1)

            lnch = [graph_part.probs[0]]
            nn_results = sess.run(
                lnch,
                {
                    self.rnn.batch_size: bi,
                    self.rnn.x_seq_lens[0]: np.asarray(seq_lens),
                    self.rnn.x_vals[0]: np.asarray(xs),
                    self.rnn.x_inds[0]: np.asarray(indexes),
                    self.rnn.x_shape[0]: np.asarray([bi, max_len])
                }
            )
            results.extend(nn_results[0])

        return results, etalon

    def __test_classification__(self, sess, key, graph_part):
        path = os.path.join(self.rnn.config['dataset_path'], f"{key}_test_dataset.pkl")
        with open(path, 'rb') as f:
            et_items = pickle.load(f)

        results, etalon = self.__get_classification_info__(sess, et_items, graph_part)
        total = len(etalon)
        total_classes = 0
        full_correct = 0
        part_correct = 0

        for index, et in enumerate(etalon):
            classes_count = et.sum()
            good_classes = np.argwhere(et == 1).ravel()
            rez_classes = np.argsort(results[index])[-classes_count:]

            total_classes += classes_count
            correct = True
            for cls in rez_classes:
                if cls in good_classes:
                    part_correct += 1
                else:
                    correct = False

            if correct:
                full_correct += 1

        full_acc = full_correct / total
        cls_correct = part_correct / total_classes

        return full_acc, cls_correct

    def __test_lemmas__(self, sess):
        path = os.path.join(self.config['dataset_path'], "lemma_test_dataset.pkl")
        with open(path, 'rb') as f:
            words = pickle.load(f)

        words = [
            word
            for word in words
            if all([c in self.config['chars'] for c in word['x_src']])
        ]

        words_to_proc = [
            (word['x_src'], word['main_cls'])
            for word in words
        ]

        rez_words = list(self.__infer_lemmas__(sess, words_to_proc))
        total = len(words)
        wrong = 0
        for index, lem in enumerate(rez_words):
            et_word = words[index]
            if lem != et_word['y_src']:
                wrong += 1

        correct = total - wrong
        acc = correct / total
        return acc

    def __test_inflect__(self, sess):
        path = os.path.join(self.config['dataset_path'], "inflect_test_dataset.pkl")
        with open(path, 'rb') as f:
            words = pickle.load(f)

        words = [
            word
            for word in words
            if all([c in self.config['chars'] for c in word['x_src']])
        ]

        words_to_proc = [
            (word['x_src'], word['main_cls'])
            for word in words
        ]

        rez_words = list(self.__infer_lemmas__(sess, words_to_proc))
        total = len(words)
        wrong = 0
        for index, lem in enumerate(rez_words):
            et_word = words[index]
            if lem != et_word['y_src']:
                wrong += 1

        correct = total - wrong
        acc = correct / total
        return acc

    def __infer_lemmas__(self, sess, words):
        wi = 0
        pbar = tqdm(total=len(words))
        while wi < len(words):
            bi = 0
            xs = []
            clss = []
            indexes = []
            seq_lens = []
            max_len = 0

            while bi < self.batch_size and wi < len(words):
                word = words[wi][0]
                cls = words[wi][1]
                for c_index, char in enumerate(word):
                    xs.append(self.chars[char])
                    indexes.append([bi, c_index])
                cur_len = len(word)
                clss.append(cls)
                if cur_len > max_len:
                    max_len = cur_len
                seq_lens.append(cur_len)
                bi += 1
                wi += 1
                pbar.update(1)

            lnch = [self.rnn.lem_graph_part.results[0]]
            results = sess.run(
                lnch,
                {
                    self.rnn.batch_size: bi,
                    self.rnn.x_seq_lens[0]: np.asarray(seq_lens),
                    self.rnn.x_vals[0]: np.asarray(xs),
                    self.rnn.x_inds[0]: np.asarray(indexes),
                    self.rnn.lem_graph_part.cls[0]: np.asarray(clss),
                    self.rnn.x_shape[0]: np.asarray([bi, max_len])
                }
            )
            for word_src in results[0]:
                yield decode_word(word_src)

    def __load_all_datasets(self, tp):
        words = []

        def load_words(type):
            path = os.path.join(self.config['dataset_path'], f"{tp}_{type}_dataset.pkl")
            with open(path, 'rb') as f:
                words.extend(pickle.load(f))

        load_words("test")
        load_words("valid")
        load_words("train")
        return words

    def get_bad_words(self):
        with tf.Session(graph=self.rnn.graph) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            self.rnn.restore(sess)
            error_words = []

            #main_cls_items = self.__load_all_datasets('main')
            #results, etalon = self.__get_classification_info__(sess, main_cls_items, self.rnn.main_graph_part)
            #for index, et in tqdm(enumerate(etalon), "Selecting main cls bad words"):
            #    classes_count = et.sum()
            #    good_classes = np.argwhere(et == 1).ravel()
            #    rez_classes = np.argsort(results[index])[-classes_count:]
#
            #    if rez_classes[0] not in good_classes:
            #        error_words.append(main_cls_items[index]['src'])
#
            #print(f'Main cls error: {len(error_words)}')
            lemma_src = self.__load_all_datasets('lemma')

            lemma_src = [
                word
                for word in lemma_src
                if all([c in self.config['chars'] for c in word['x_src']])
            ]

            words_to_proc = [
                (word['x_src'], word['main_cls'])
                for word in lemma_src
            ]

            lemma_results = list(self.__infer_lemmas__(sess, words_to_proc))
            for index, lem in tqdm(enumerate(lemma_results), desc="Selecting lemma bad words"):
                et_word = lemma_src[index]
                if lem != et_word['y_src']:
                    error_words.append(et_word['x_src'])

            print(f'Total error: {len(error_words)}')
            error_words = list(set(error_words))
            print(f'Total unique error: {len(error_words)}')
            return error_words

    def get_test_lemmas(self, words):
        with tf.Session(graph=self.rnn.graph) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            latest_checkpoint = tf.train.latest_checkpoint(self.rnn.save_path)
            self.rnn.saver.restore(sess, latest_checkpoint)
            return list(self.__infer_lemmas__(sess, words))


if __name__ == "__main__":
    tester = Tester()
    tester.test()
