import re
import os
import gzip
import yaml
import tqdm
import pickle
import logging
from lxml import etree
from xml.etree.ElementTree import ElementTree
from utils import get_flat_words, config

CONFIG = config()
NAR_REG = re.compile("\d+-.*")
RANDOM_SEED = 1917
DATASET_PATH = CONFIG['dataset_path']
REZ_PATHS = CONFIG['publish_dictionary_paths']
DICT_WORDS_PATH = CONFIG['dict_words_path']
NOT_DICT_WORDS_PATH = CONFIG['dataset_words_path']
MAX_WORD_SIZE = CONFIG['max_word_size']
DICT_POST_TYPES = CONFIG['dict_post_types']
GRAMMEMES_TYPES = CONFIG['grammemes_types']
SOGL_CHARS = [
    'б',
    'в',
    'г',
    'д',
    'ж',
    'з',
    'й',
    'к',
    'л',
    'м',
    'н',
    'п',
    'р',
    'с',
    'т',
    'ф',
    'х',
    'ц',
    'ч',
    'ш',
    'щ'
]
GLASN_CHARS = ['а', 'о', 'и', 'е', 'ё', 'э', 'ы', 'у', 'ю', 'я']


CLASSES_INDEX_DICT = {
    cls: GRAMMEMES_TYPES[gram]['classes'][cls]['index']
    for gram in sorted(GRAMMEMES_TYPES, key=lambda x: GRAMMEMES_TYPES[x]['index'])
    for cls in GRAMMEMES_TYPES[gram]['classes']
}

POST_POWER_DICT = {}
for key in DICT_POST_TYPES:
    POST_POWER_DICT[key] = DICT_POST_TYPES[key]['power'] if 'power' in DICT_POST_TYPES[key] else 1

p_dic = GRAMMEMES_TYPES['post']['classes']
for key in p_dic:
    POST_POWER_DICT[key] = p_dic[key]['power'] if 'power' in p_dic[key] else 1



class Word:
    def __init__(self, word):
        self.word = word
        self.index = 0
        self.text = word['text']
        self.length = len(word['text'])
        self.grams = []
        self.add_gram(word)

    def add_gram(self, word):
        grams = [word[key] if key in word else None for key in sorted(GRAMMEMES_TYPES, key=lambda k: GRAMMEMES_TYPES[k]['index'])]
        power = [CLASSES_INDEX_DICT[key] if key and key in CLASSES_INDEX_DICT else 1000 for key in grams]
        power.insert(0, -POST_POWER_DICT[word['post']])
        power = tuple(power)
        self.grams.append({
            'lemma': word['lemma'] if 'lemma' in word else None,
            'gram': ",".join([gram if gram else '' for gram in grams]),
            'power': power
        })

    def next(self):
        self.index += 1

    def is_finished(self):
        return self.index == self.length

    def current(self):
        return self.text[self.index]

    def get_grams(self):
        items = sorted(self.grams, key=lambda g: g['power'])
        return items

    def __repr__(self):
        return f"{self.text} - {self.text[self.index:]}"



def get_numbers():
    with open('numb.yml') as f:
        numbr_dic = yaml.load(f)

    def get_nar_end(text):
        end = text[-1]
        if text[-2] in SOGL_CHARS and text[-1] in GLASN_CHARS:
            end = text[-2:]

        return end

    numbers = []
    nar_dict = {}
    for n_key in numbr_dic:
        for t in numbr_dic[n_key]:
            lemma_text = None
            lemma_number_text = None
            for index, item in enumerate(numbr_dic[n_key][t]):
                if index == 0:
                    lemma_text = item['text']

                item['post'] = 'numb'
                item['lemma'] = lemma_text
                numbers.append(Word(item))

                if t == 'p':
                    end = get_nar_end(item['text'])
                    if index == 0:
                        lemma_number_text = end

                    text = f"{n_key}-{end}"
                    item['lemma'] = lemma_number_text
                    item['text'] = text
                    if text in nar_dict:
                        nar_dict[text].add_gram(item)
                    else:
                        word = Word(item)
                        nar_dict[text] = word
                        numbers.append(word)

    return numbers

def release_tree_dict():

    with open(DICT_WORDS_PATH, 'rb') as f:
        words = [Word(s_word) for s_word in pickle.load(f)]

    with open(NOT_DICT_WORDS_PATH, 'rb') as f:
        not_dict_words = pickle.load(f)
        io_words = [io for io in not_dict_words if 'ё' in io]
        for io_word in io_words:
            io_items = not_dict_words[io_word]
            not_io_word = io_word.replace('ё', 'е')
            not_io_items = []
            for item in io_items:
                not_io_item = dict(item)
                not_io_item['lemma'] = not_io_item['lemma'].replace('ё', 'е')
                not_io_item['text'] = not_io_item['text'].replace('ё', 'е')
                not_io_items.append(not_io_item)

            not_dict_words[not_io_word] = not_io_items


    for numb in get_numbers():
        words.append(numb)




    dwords_dic = {word.text: word for word in words}
    for nd_word in tqdm.tqdm(not_dict_words, desc='Looking for duplicates'):
        if nd_word in dwords_dic and not NAR_REG.match(nd_word):
            for gram in not_dict_words[nd_word]:
                dwords_dic[nd_word].add_gram(gram)

    with open("wrong_words.pkl", 'rb') as f:
        wrongs = pickle.load(f)

    for w_word in tqdm.tqdm(wrongs, desc='Processing wrongs'):
        if not NAR_REG.match(w_word) and w_word in not_dict_words:
            for gram in not_dict_words[w_word]:
                if w_word in dwords_dic:
                    dwords_dic[w_word].add_gram(gram)
                else:
                    dwords_dic[w_word] = Word(gram)

    root = etree.Element('Tree')
    cur_items = [(root, words)]

    while len(cur_items) != 0:
        new_cur_items = []
        for par_el, par_words in cur_items:

            c_dic = {}
            for w in par_words:
                if w.current() not in c_dic:
                    c_dic[w.current()] = []
                c_dic[w.current()].append(w)

            for c in c_dic:
                words = c_dic[c]
                leaf = etree.Element('L')
                leaf.set('c', c)
                not_fin_words = []
                for word in words:
                    word.next()
                    if word.is_finished():
                        leaf.set('t', word.text)
                        for gram in word.get_grams():
                            gram_el = etree.Element('G')
                            gram_el.set('v', gram['gram'])
                            if gram['lemma']:
                                gram_el.set('l', gram['lemma'])
                            leaf.append(gram_el)
                    else:
                        not_fin_words.append(word)

                par_el.append(leaf)
                if len(not_fin_words)>0:
                    new_cur_items.append((leaf, not_fin_words))

        cur_items = new_cur_items

    tree = ElementTree(root)
    for path in REZ_PATHS:
        path = os.path.join(path, "tree_dict.xml.gz")
        with gzip.open(path, 'wb+') as f:
            tree.write(f, xml_declaration=True, encoding='utf-8')

    logging.info("Tree dictionary released")
    logging.info(f"Words count {len(dwords_dic)}")