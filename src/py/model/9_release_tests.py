
import numpy as np

from tester import Tester


def __release_test_results__(self, tester):
    results = tester.test()
    for path in self.test_result_paths:
        with open(path, 'w+') as f:
            f.write(results)


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


testr = Tester()
self.__release_test_results__(testr)
self.__build_bad_words__(testr)