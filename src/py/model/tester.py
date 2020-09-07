import os, pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from model import RNN
from utils import CONFIG, decode_word, load_datasets


class Tester:
    def __init__(self):
        self.config = CONFIG
        self.config['graph_part_configs']['lemm']['use_cls_placeholder'] = True
        self.rnn = RNN(True)
        self.chars = {c: index for index, c in enumerate(self.config['chars'])}
        self.batch_size = 65536
        self.show_bad_items = False

    def test(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        results = []
        with tf.Session(config=config, graph=self.rnn.graph) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            self.rnn.restore(sess)
            for gram in self.rnn.gram_keys:
                full_cls_acc, part_cls_acc, _ = self.__test_classification__(sess, gram, self.rnn.gram_graph_parts[gram], 'test')
                result = f"{gram}. full_cls_acc: {full_cls_acc}; part_cls_acc: {part_cls_acc}"
                results.append(result)
                tqdm.write(result)

            full_cls_acc, part_cls_acc, _ = self.__test_classification__(sess, 'main', self.rnn.main_graph_part, 'test')
            result = f"main. full_cls_acc: {full_cls_acc}; part_cls_acc: {part_cls_acc}"
            results.append(result)
            tqdm.write(result)
            lemm_acc, _ = self.__test_lemmas__(sess, 'test')
            result = f"lemma_acc: {lemm_acc}"
            tqdm.write(result)
            results.append(result)
            inflect_acc, _ = self.__test_inflect__(sess, 'test')
            result = f"inflect_acc: {inflect_acc}"
            tqdm.write(result)
            results.append(result)
            tqdm.write(result)

        return "\n".join(results)

    def __get_classification_items__(self, sess, items, graph_part):
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

    def __get_lemma_items__(self, sess, items):
        wi = 0
        pbar = tqdm(total=len(items))
        while wi < len(items):
            bi = 0
            xs = []
            clss = []
            indexes = []
            seq_lens = []
            max_len = 0

            while bi < self.batch_size and wi < len(items):
                item = items[wi]
                word = item['x_src']
                x_cls = item['main_cls']
                for c_index, char in enumerate(word):
                    xs.append(self.chars[char])
                    indexes.append([bi, c_index])
                cur_len = len(word)
                clss.append(x_cls)
                if cur_len > max_len:
                    max_len = cur_len
                seq_lens.append(cur_len)
                bi += 1
                wi += 1
                pbar.update(1)

            lnch = [self.rnn.lem_result]
            results = sess.run(
                lnch,
                {
                    self.rnn.batch_size: bi,
                    self.rnn.x_seq_lens[0]: np.asarray(seq_lens),
                    self.rnn.x_vals[0]: np.asarray(xs),
                    self.rnn.x_inds[0]: np.asarray(indexes),
                    self.rnn.lem_class_pl: np.asarray(clss),
                    self.rnn.x_shape[0]: np.asarray([bi, max_len])
                }
            )
            for word_src in results[0]:
                yield decode_word(word_src[0])

    def __get_inflect_items__(self, sess, items):
        wi = 0
        pbar = tqdm(total=len(items))
        while wi < len(items):
            bi = 0
            xs = []
            x_clss = []
            y_clss = []
            indexes = []
            seq_lens = []
            max_len = 0

            while bi < self.batch_size and wi < len(items):
                item = items[wi]
                word = item['x_src']
                x_cls = item['x_cls']
                y_cls = item['y_cls']
                for c_index, char in enumerate(word):
                    xs.append(self.chars[char])
                    indexes.append([bi, c_index])
                cur_len = len(word)
                x_clss.append(x_cls)
                y_clss.append(y_cls)
                if cur_len > max_len:
                    max_len = cur_len
                seq_lens.append(cur_len)
                bi += 1
                wi += 1
                pbar.update(1)

            lnch = [self.rnn.inflect_graph_part.results[0]]
            results = sess.run(
                lnch,
                {
                    self.rnn.batch_size: bi,
                    self.rnn.x_seq_lens[0]: np.asarray(seq_lens),
                    self.rnn.x_vals[0]: np.asarray(xs),
                    self.rnn.x_inds[0]: np.asarray(indexes),
                    self.rnn.inflect_graph_part.x_cls[0]: np.asarray(x_clss),
                    self.rnn.inflect_graph_part.y_cls[0]: np.asarray(y_clss),
                    self.rnn.x_shape[0]: np.asarray([bi, max_len])
                }
            )

            for word_src in results[0]:
                yield decode_word(word_src)

    def __test_classification__(self, sess, key, graph_part, *ds_types):
        et_items = load_datasets(key, *ds_types)
        results, etalon = self.__get_classification_items__(sess, et_items, graph_part)
        total = len(etalon)
        total_classes = 0
        full_correct = 0
        part_correct = 0
        bad_items = []

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
            else:
                bad_items.append((et_items[index], rez_classes))

        full_acc = full_correct / total
        cls_correct = part_correct / total_classes
        return full_acc, cls_correct, bad_items

    def __test_lemmas__(self, sess, *ds_types):
        good_items = load_datasets("lemma", *ds_types)
        good_items = [
            word
            for word in good_items
            if all([c in self.config['chars'] for c in word['x_src']])
        ]
        results = list(self.__get_lemma_items__(sess, good_items))

        bad_words = []
        total = len(good_items)
        wrong = 0
        for index, rez in enumerate(results):
            et_word = good_items[index]
            if rez != et_word['y_src']:
                wrong += 1
                bad_words.append((et_word, rez))

        correct = total - wrong
        acc = correct / total
        return acc, bad_words

    def __test_inflect__(self, sess, *ds_types):
        good_items = load_datasets("inflect", *ds_types)
        good_items = [
            word
            for word in good_items
            if all([c in self.config['chars'] for c in word['x_src']])
        ]

        bad_items = []
        rez_words = list(self.__get_inflect_items__(sess, good_items))
        total = len(good_items)
        wrong = 0
        for index, rez in enumerate(rez_words):
            et_word = good_items[index]
            if rez != et_word['y_src']:
                wrong += 1
                bad_items.append((et_word, rez))

        correct = total - wrong
        acc = correct / total
        return acc, bad_items


if __name__ == "__main__":
    tester = Tester()
    tester.test()
