import string
import re
import os
import gzip
import yaml
import tqdm
import pickle
import logging
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


def release_tree_dict(with_wrongs=True):

    with open(DICT_WORDS_PATH, 'rb') as f:
        words = pickle.load(f)
        words = [Word(s_word) for s_word in words]

    with open(NOT_DICT_WORDS_PATH, 'rb') as f:
        not_dict_words = pickle.load(f)
        io_words = [io for io in not_dict_words if 'ё' in io]
        for io_word in io_words:
            io_items = not_dict_words[io_word]
            not_io_word = io_word.replace('ё', 'е')
            not_io_items = []
            for item in io_items:
                not_io_item = dict(item)
                if 'lemma' in not_io_item:
                    not_io_item['lemma'] = not_io_item['lemma'].replace('ё', 'е')
                not_io_item['text'] = not_io_item['text'].replace('ё', 'е')
                not_io_items.append(not_io_item)

            not_dict_words[not_io_word] = not_io_items

    for numb in get_numbers():
        words.append(numb)

    if with_wrongs:
        with open("wrong_words.pkl", 'rb') as f:
            wrongs = set(pickle.load(f))
    else:
        wrongs = set()

    dwords_dic = {word.text: word for word in words}
    for nd_word in tqdm.tqdm(not_dict_words, desc='Looking for duplicates'):
        if nd_word in dwords_dic and not NAR_REG.match(nd_word):
            for gram in not_dict_words[nd_word]:
                dwords_dic[nd_word].add_gram(gram)
        elif nd_word in wrongs:
            items = not_dict_words[nd_word]
            word = Word(items[0])
            del items[0]
            for item in items:
                word.add_gram(item)

            words.append(word)


    ind_keys = string.ascii_lowercase + string.ascii_uppercase
    index_dict = {}
    cur_index = 0
    txt = []
    for word in words:
        txt.append(word.text)
        txt.append('\t')
        for ind, g in enumerate(word.get_grams()):
            if 'lemma' in g and g['lemma']:
                txt.append(g['lemma'])
            txt.append(':')
            for g_ind, val in enumerate(g['gram'].split(',')):
                if val and val not in index_dict:
                    index_dict[val] = ind_keys[cur_index]
                    cur_index+=1

                if val:
                    txt.append(index_dict[val])

                if g_ind != len(GRAMMEMES_TYPES):
                    txt.append(',')

            if ind != len(word.grams)-1:
                txt.append(';')

        txt.append('\n')

    txt.insert(0, "\n")
    for key in index_dict:
        txt.insert(0, f"{index_dict[key]}={key}\n")

    txt = "".join(txt)
    rez = txt.encode('utf-8')

    for path in REZ_PATHS:
        path = os.path.join(path, "dict.txt.gz")
        with gzip.open(path, 'wb+') as f:
            f.write(rez)






    logging.info("Tree dictionary released")
    logging.info(f"Words count {len(words)}")

if __name__ == "__main__":
    release_tree_dict()