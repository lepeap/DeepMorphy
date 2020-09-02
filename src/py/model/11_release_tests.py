import os
import string
import pickle
import numpy as np
from utils import CONFIG, RANDOM, load_datasets

from lxml import etree
from xml.etree.ElementTree import ElementTree


GRAM_TYPES = CONFIG['grammemes_types']
ROOT = CONFIG['publish_tests_path']
DICT_WORDS_PATH = CONFIG['dict_words_path']
NMB_DATA_PATH = CONFIG['numb_data_path']
with open(NMB_DATA_PATH, 'rb') as f:
    numb_data = pickle.load(f)

with open(CONFIG['tags_path'], 'rb') as f:
    tags = pickle.load(f)
    is_lemma_dict = {tags[key]['i']: tags[key]['l'] for key in tags}
    tag_index_order = {tags[tag]['i']: tags[tag]['o'] for tag in tags}

def release_gram_tests(items, key, cls_dic, result_path):
    root = etree.Element('Tests')
    for word in items:
        y = np.argwhere(word['y'] == 1).ravel()
        y = ';'.join([cls_dic[index] for index in y])
        test = etree.Element("T")
        test.set('x', word['src'])
        test.set('y', y)
        root.append(test)

    rez_path = os.path.join(result_path, f'{key}.xml')
    tree = ElementTree(root)
    with open(rez_path, 'wb+') as f:
        tree.write(f, xml_declaration=True, encoding='utf-8')


def release_main_tests(items, result_path, y_is_index=True):
    root = etree.Element('Tests')
    for word in items:
        if y_is_index:
            y = word['y']
        else:
            y = np.argwhere(word['y'] == 1).ravel()
        y = ';'.join([str(index) for index in y])
        test = etree.Element("T")
        test.set('x', word['src'])
        test.set('y', y)
        root.append(test)

    rez_path = os.path.join(result_path, f'main.xml')
    tree = ElementTree(root)
    with open(rez_path, 'wb+') as f:
        tree.write(f, xml_declaration=True, encoding='utf-8')


def release_lemma_tests(items, result_path):
    root = etree.Element('Tests')
    for word in items:
        test = etree.Element("T")
        test.set('x', word['x_src'])
        test.set('x_c', str(word['main_cls']))
        test.set('y', word['y_src'])
        root.append(test)

    rez_path = os.path.join(result_path, 'lemma.xml')
    tree = ElementTree(root)
    with open(rez_path, 'wb+') as f:
        tree.write(f, xml_declaration=True, encoding='utf-8')


def release_inflect_tests(items, result_path):
    root = etree.Element('Tests')
    for word in items:
        test = etree.Element("T")
        test.set('x', word['x_src'])
        test.set('x_c', str(word['x_cls']))
        test.set('y', word['y_src'])
        test.set('y_c', str(word['y_cls']))
        root.append(test)

    rez_path = os.path.join(result_path, 'inflect.xml')
    tree = ElementTree(root)
    with open(rez_path, 'wb+') as f:
        tree.write(f, xml_declaration=True, encoding='utf-8')


def release_pure_nn_tests():
    res_path = os.path.join(ROOT, 'OnlyNn')
    for gram in GRAM_TYPES:
        items = load_datasets(gram, 'test')
        cls = GRAM_TYPES[gram]['classes']
        dic = {cls[g_key]['index']: g_key for g_key in cls}
        release_gram_tests(items, gram, dic, res_path)

    release_main_tests(load_datasets('main', 'test'), res_path, False)
    release_lemma_tests(load_datasets('lemma', 'test'), res_path)
    release_inflect_tests(load_datasets('inflect', 'test'), res_path)


def merge_same_main(items):
    rez_dict = {}
    for item in items:
        if item['src'] not in rez_dict:
            rez_dict[item['src']] = []

        rez_dict[item['src']].append(item)

    rez_list = []
    for text in rez_dict:
        ys = [item['y'] for item in rez_dict[text]]
        rez_list.append(dict(src=text, y=ys))
    return rez_list


def release_dictionary_tests():
    res_path = os.path.join(ROOT, 'Dict')
    with open(DICT_WORDS_PATH, 'rb') as f:
        words = pickle.load(f)

    lexeme_dict = {}
    for word in words:
        if word['id'] not in lexeme_dict:
            lexeme_dict[word['id']] = []

        lexeme_dict[word['id']].append(word)

    main = []
    inflect = []
    lemmas = []
    for word_id in lexeme_dict:
        lexeme_words = lexeme_dict[word_id]
        for item in lexeme_words:
            main.append(dict(src=item['text'], y=item['main']))

        for i in range(0, len(lexeme_words) - 2):
            main_word = lexeme_words[i]
            for j in range(i, len(lexeme_words) - 1):
                to_word = lexeme_words[j]
                inflect.append(dict(
                    x_src=main_word['text'],
                    x_cls=main_word['main'],
                    y_src=to_word['text'],
                    y_cls=to_word['main'],
                    id=word_id
                ))

        for word in lexeme_words:
            lemmas.append(dict(
                x_src=word['text'],
                main_cls=word['main'],
                y_src=word['lemma'] if 'lemma' in word else word['text']
            ))

    main = merge_same_main(main)
    release_main_tests(main, res_path)
    release_lemma_tests(lemmas, res_path)
    release_inflect_tests(inflect, res_path)


def release_numb_tests():
    res_path = os.path.join(ROOT, 'Numb')
    main = []
    inflect = []
    lemmas = []
    for val in numb_data['numbers']:
        n_el = etree.Element("N")
        n_el.set('v', str(val))
        for tp in numb_data['numbers'][val]:
            if tp == 'nar_end' or tp == 'lemma':
                continue

            items = numb_data['numbers'][val][tp]
            lemma, _ = items[0]
            for text, index in items:
                main.append(dict(src=text, y=index))
                lemmas.append(dict(
                    x_src=text,
                    main_cls=index,
                    y_src=lemma
                ))

            un_cls_ids = []
            rez_items = []
            for item in items:
                if item[1] in un_cls_ids:
                    continue

                un_cls_ids.append(item[1])
                rez_items.append(item)
            items = rez_items

            for i in range(0, len(items) - 2):
                main_text, main_index = items[i]
                for j in range(i, len(items) - 1):
                    to_text, to_index = items[j]
                    inflect.append(dict(
                        x_src=main_text,
                        x_cls=main_index,
                        y_src=to_text,
                        y_cls=to_index,
                        id=f"{val}{tp}"
                    ))

    main = merge_same_main(main)
    release_main_tests(main, res_path)
    release_lemma_tests(lemmas, res_path)
    release_inflect_tests(inflect, res_path)


def release_nar_numb_tests():
    res_path = os.path.join(ROOT, 'NarNumb')
    main = []
    inflect = []
    lemmas = []
    for val in numb_data['numbers']:
        n_el = etree.Element("N")
        n_el.set('v', str(val))
        items = numb_data['numbers'][val]['nar_end']
        lemma_id = numb_data['numbers'][val]['p'][0][1]
        lemma = f"{val}-{items[lemma_id]}"
        for index in items:
            text = f"{val}-{items[index]}"
            main.append(dict(src=text, y=index))
            lemmas.append(dict(
                x_src=text,
                main_cls=index,
                y_src=lemma
            ))

        ids = list(items.keys())
        for i in range(0, len(ids) - 2):
            main_index = ids[i]
            main_text = items[main_index]
            main_text = f"{val}-{main_text}"
            for j in range(i, len(items) - 1):
                to_index = ids[j]
                to_text = items[to_index]
                to_text = f"{val}-{to_text}"
                inflect.append(dict(
                    x_src=main_text,
                    x_cls=main_index,
                    y_src=to_text,
                    y_cls=to_index,
                    id=f"{val}nar"
                ))

    main = merge_same_main(main)
    release_main_tests(main, res_path)
    release_lemma_tests(lemmas, res_path)
    release_inflect_tests(inflect, res_path)


def release_reg_tests():
    res_path = os.path.join(ROOT, 'Reg')
    int_tag = None
    romn_tag = None
    unkn_tag = None
    punct_tag = None

    for tag in tags:
        if 'int' in tag:
            int_tag = tags[tag]['i']
        elif 'romn' in tag:
            romn_tag = tags[tag]['i']
        elif 'unkn' in tag:
            unkn_tag = tags[tag]['i']
        elif 'punct' in tag:
            punct_tag = tags[tag]['i']

    main = []
    puncts = ['.', ',', '?', '!', '_', '"', '(', ')', ':', ';', '-']
    for p in puncts:
        main.append(dict(src=p, y=punct_tag))

        text = [p]
        for _ in range(1, RANDOM.randint(2, 5)):
            text.append(puncts[RANDOM.randint(0, len(puncts)-1)])

        text = ''.join(text)
        main.append(dict(src=text, y=punct_tag))

    for i in range(100):
        val = RANDOM.randint(0, 1000000)
        main.append(dict(src=str(val), y=int_tag))

    roms = ['i', 'iii', 'iv', 'iv', 'c', 'd', 'm', 'md', 'mi']
    for rom in roms:
        main.append(dict(src=rom, y=romn_tag))
        main.append(dict(src=rom.upper(), y=romn_tag))

    unkn = ['test', 'sdasdas', 'home']
    for v in unkn:
        main.append(dict(src=v, y=unkn_tag))
        main.append(dict(src=v.upper(), y=unkn_tag))

    lemmas = [dict(x_src=item['src'], main_cls=item['y'], y_src=item['src']) for item in main]
    inflect = [dict(x_src=item['src'], x_cls=item['y'], y_src=item['src'], y_cls=item['y'], id=item['src']) for item in main]

    main = merge_same_main(main)
    release_main_tests(main, res_path)
    release_lemma_tests(lemmas, res_path)
    release_inflect_tests(inflect, res_path)


release_reg_tests()
release_nar_numb_tests()
release_numb_tests()
release_dictionary_tests()
release_pure_nn_tests()