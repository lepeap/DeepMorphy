import os
import pickle
import numpy as np
from utils import CONFIG, load_datasets

from lxml import etree
from xml.etree.ElementTree import ElementTree


GRAM_TYPES = CONFIG['grammemes_types']
TEST_RESULT_PATHS = CONFIG['publish_test_paths']


def release_gram_tests(items, key, cls_dic):
    root = etree.Element('Tests')
    for word in items:
        y = np.argwhere(word['y'] == 1).ravel()
        y = ';'.join([cls_dic[index] for index in y])
        test = etree.Element("T")
        test.set('x', word['src'])
        test.set('y', y)
        root.append(test)

    for dir_path in TEST_RESULT_PATHS:
        rez_path = os.path.join(dir_path, f'{key}.xml')
        tree = ElementTree(root)
        with open(rez_path, 'wb+') as f:
            tree.write(f, xml_declaration=True, encoding='utf-8')


def release_main_tests():
    items = load_datasets('main', 'test')
    root = etree.Element('Tests')
    for word in items:
        y = np.argwhere(word['y'] == 1).ravel()
        y = ';'.join([str(index) for index in y])
        test = etree.Element("T")
        test.set('x', word['src'])
        test.set('y', y)
        root.append(test)

    for dir_path in TEST_RESULT_PATHS:
        rez_path = os.path.join(dir_path, f'main.xml')
        tree = ElementTree(root)
        with open(rez_path, 'wb+') as f:
            tree.write(f, xml_declaration=True, encoding='utf-8')


def release_lemma_tests():
    items = load_datasets('lemma', 'test')

    root = etree.Element('Tests')
    for word in items:
        test = etree.Element("T")
        test.set('x', word['x_src'])
        test.set('x_c', str(word['main_cls']))
        test.set('y', word['y_src'])
        root.append(test)

    for dir_path in TEST_RESULT_PATHS:
        rez_path = os.path.join(dir_path, 'lemma.xml')
        tree = ElementTree(root)
        with open(rez_path, 'wb+') as f:
            tree.write(f, xml_declaration=True, encoding='utf-8')


def release_inflect_tests():
    items = load_datasets('inflect', 'test')

    root = etree.Element('Tests')
    for word in items:
        test = etree.Element("T")
        test.set('x', word['x_src'])
        test.set('x_c', str(word['x_cls']))
        test.set('y', word['y_src'])
        test.set('y_c', str(word['y_cls']))
        root.append(test)

    for dir_path in TEST_RESULT_PATHS:
        rez_path = os.path.join(dir_path, 'inflect.xml')
        tree = ElementTree(root)
        with open(rez_path, 'wb+') as f:
            tree.write(f, xml_declaration=True, encoding='utf-8')


for gram in GRAM_TYPES:
    items = load_datasets(gram, 'test')
    cls = GRAM_TYPES[gram]['classes']
    dic = {cls[g_key]['index']: g_key for g_key in cls}
    release_gram_tests(items, gram, dic)

release_main_tests()
release_lemma_tests()
release_inflect_tests()
