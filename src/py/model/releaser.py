import os
import pickle
import logging
import numpy as np
from shutil import copyfile
from model import RNN
from utils import config
from lxml import etree
from xml.etree.ElementTree import ElementTree
from tester import Tester


class Releaser:
    def __init__(self):
        self.config = config()
        self.dataset_path = self.config['dict_path']
        self.model_key = self.config['model_key']
        self.chars = self.config['chars']
        self.gram_types = self.config['grammemes_types']
        self.rnn = RNN(True)
        self.pd_publish_paths = [
            os.path.join(path, f"frozen_model_{self.model_key}.pb")
            for path in self.config['publish_net_paths']
        ]
        self.xml_publish_paths = [
            os.path.join(path, f"release_{self.model_key}.xml")
            for path in self.config['publish_net_paths']
        ]
        self.xml_gram_paths = [
            os.path.join(path, "grams.xml")
            for path in self.config['publish_gramm_paths']
        ]
        self.test_result_paths = [
            os.path.join(path, "test_info.txt")
            for path in self.config['test_results_paths']
        ]
        self.publish_dataset_info_paths = [
            os.path.join(path, "dataset_info.txt")
            for path in self.config['publish_dataset_info_paths']
        ]
        self.tests_results_paths = self.config['publish_test_paths']
        self.classes_dic = self.config['main_classes']
        self.rev_classes_dic = {
            self.classes_dic[key]: ",".join([key for key in list(key) if key is not None])
            for key in self.classes_dic
        }


    def release_model(self):
        pd_release_path, gram_ops, out_ops = self.rnn.release()
        for path in self.pd_publish_paths:
            copyfile(pd_release_path, path)

        self.__release_test_files__()
        self.__release_dataset_info__()
        self.__release_model_xml__(out_ops, gram_ops)
        self.__release_grams_xml__()
        tester = Tester()
        self.__release_test_results__(tester)
        self.__build_bad_words__(tester)


    def __release_model_xml__(self, out_ops, gram_ops):
        root = etree.Element('Root')
        for key in out_ops:
            root.set(key, out_ops[key])

        chars_el = etree.Element('Chars')
        chars_el.set("start_char", str(self.config['start_token']))
        chars_el.set("end_char", str(self.config['end_token']))

        for index, value in enumerate(self.chars):
            char_el = etree.Element("Char")
            char_el.set('index', str(index))
            char_el.set('value', value)
            chars_el.append(char_el)

        root.append(chars_el)
        grams_el = etree.Element('Grams')
        for gram in self.gram_types:
            gram_el = etree.Element("G")
            gram_el.set('key', gram)
            gram_el.set('op', gram_ops[gram]['prob'])
            grams_el.append(gram_el)

        root.append(grams_el)
        classes_el = etree.Element('Classes')
        for cls in self.classes_dic:
            cls_el = etree.Element("C")
            cls_el.set('i', str(self.classes_dic[cls]))
            cls_el.set('v', ",".join([key if key is not None else '' for key in cls]))

            classes_el.append(cls_el)

        root.append(classes_el)
        tree = ElementTree(root)
        for path in self.xml_publish_paths:
            with open(path, 'wb+') as f:
                tree.write(f, xml_declaration=True, encoding='utf-8')

        logging.info("Model released")

    def __release_grams_xml__(self):
        nn_types = self.config['grammemes_types']
        dict_post_types = self.config['dict_post_types']
        other_types = self.config['other_post_types']
        root = etree.Element('Grams')
        for gram in nn_types:
            gram_el = etree.Element("G")
            gram_el.set('index', str(nn_types[gram]['index']))
            gram_el.set('key_en', gram)
            gram_el.set('key_ru', nn_types[gram]['key_ru'])
            root.append(gram_el)
            gr_dic = nn_types[gram]['classes']
            for key_en in gr_dic:
                item = gr_dic[key_en]
                cls_el = etree.Element("C")
                cls_el.set('key_en', key_en)
                cls_el.set('key_ru', str(item['key_ru']))
                cls_el.set('nn_index', str(item['index']))
                gram_el.append(cls_el)

            if gram == "post":
                for key_en in dict_post_types:
                    item = dict_post_types[key_en]
                    cls_el = etree.Element("C")
                    cls_el.set('key_en', key_en)
                    cls_el.set('key_ru', str(item['key_ru']))
                    gram_el.append(cls_el)

                for key_en in other_types:
                    item = other_types[key_en]
                    cls_el = etree.Element("C")
                    cls_el.set('key_en', key_en)
                    cls_el.set('key_ru', str(item['key_ru']))
                    gram_el.append(cls_el)

        tree = ElementTree(root)
        for path in self.xml_gram_paths:
            with open(path, 'wb+') as f:
                tree.write(f, xml_declaration=True, encoding='utf-8')

    def __release_dataset_info__(self):
        doc = etree.iterparse(self.dataset_path, events=('start', 'end'))
        itr = iter(doc)
        event, element = next(itr)
        while not (event == 'start' and element.tag == 'dictionary'):
            pass

        version = element.attrib['version']
        revision = element.attrib['revision']
        for path in self.publish_dataset_info_paths:
            with open(path, 'w+') as f:
                f.write(f"dictionary\nversion={version}\nrevision={revision}")

    def __release_test_results__(self, tester):
        results = tester.test()
        for path in self.test_result_paths:
            with open(path, 'w+') as f:
                f.write(results)

    def __build_bad_words__(self, tester):
        words = []

        def load_words(type):
            path = os.path.join(tester.config['dataset_path'], f"lemma_{type}_dataset.pkl")
            with open(path, 'rb') as f:
                words.extend(pickle.load(f))

        load_words("test")
        load_words("valid")
        load_words("train")

        words = [
            word
            for word in words
            if all([c in tester.config['chars'] for c in word['x_src']])
        ]

        words_to_parse = [
            (word['x_src'], word['main_cls'])
            for word in words
        ]

        lemmas = tester.get_test_lemmas(words_to_parse)

        wrong_words = []
        for index, lem in enumerate(lemmas):
            et_word = words[index]
            if lem != et_word['y_src']:
                wrong_words.append(et_word['x_src'])

        print(f"Wrong lemmas count: {len(wrong_words)}")
        with open(os.path.join("wrong_words.pkl"), 'wb+') as f:
            pickle.dump(wrong_words, f)

    def __release_test_files__(self):
        for gram in self.gram_types:
            cls = self.gram_types[gram]['classes']
            dic = {cls[g_key]['index']: g_key for g_key in cls}
            self.__release_cls_tests__(gram, dic)

        self.__release_cls_tests__('main', self.rev_classes_dic)
        self.__release_lemma_tests__()

    def __release_cls_tests__(self, key, cls_dic):
        path = os.path.join(self.config['dataset_path'], f"{key}_test_dataset.pkl")
        with open(path, 'rb') as f:
            words = pickle.load(f)

        root = etree.Element('Tests')
        for word in words:
            y = np.argwhere(word['y'] == 1).ravel()
            y = ';'.join([cls_dic[index] for index in y])
            test = etree.Element("T")
            test.set('x', word['src'])
            test.set('y', y)
            root.append(test)

        for dir_path in self.tests_results_paths:
            rez_path = os.path.join(dir_path, f'{key}.xml')
            tree = ElementTree(root)
            with open(rez_path, 'wb+') as f:
                tree.write(f, xml_declaration=True, encoding='utf-8')

    def __release_lemma_tests__(self):
        path = os.path.join(self.config['dataset_path'], "lemma_test_dataset.pkl")
        with open(path, 'rb') as f:
            words = pickle.load(f)

        root = etree.Element('Tests')
        for word in words:
            test = etree.Element("T")
            test.set('x', word['x_src'])
            test.set('y', word['y_src'])
            root.append(test)

        for dir_path in self.tests_results_paths:
            rez_path = os.path.join(dir_path, 'lem.xml')
            tree = ElementTree(root)
            with open(rez_path, 'wb+') as f:
                tree.write(f, xml_declaration=True, encoding='utf-8')




if __name__ == "__main__":
    tester = Releaser()
    tester.release_model()