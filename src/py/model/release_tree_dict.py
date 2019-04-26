import yaml
import tqdm
import pickle
import logging
from lxml import etree
from xml.etree.ElementTree import ElementTree
from utils import get_flat_words, config

CONFIG = config()
RANDOM_SEED = 1917
DATASET_PATH = CONFIG['dataset_path']
REZ_PATH = CONFIG['publish_dictionary_path']
WORDS_PATH = CONFIG['words_path']
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


with open(WORDS_PATH, 'rb') as f:
    words = pickle.load(f)

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
        return word.index == word.length

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

    numbers = []
    nar_dict = {}
    for n_key in numbr_dic:
        for t in numbr_dic[n_key]:
            for item in numbr_dic[n_key][t]:
                item['post'] = 'numb'
                numbers.append(Word(item))

                if t == 'p':

                    end = item['text'][-1]
                    if item['text'][-2] in SOGL_CHARS and item['text'][-1] in GLASN_CHARS:
                        end = item['text'][-2:]

                    text = f"{n_key}-{end}"
                    if 'lemma' in item:
                        del item['lemma']
                    item['text'] = text
                    if text in nar_dict:
                        nar_dict[text].add_gram(item)
                    else:
                        word = Word(item)
                        nar_dict[text] = word
                        numbers.append(word)

    return numbers



dict_words = [
    word for word in words
    if word['lemma']['post'] in DICT_POST_TYPES
       and word['lemma']['post']!='numb'
]
dict_words = [Word(word) for word in get_flat_words(dict_words)]
for numb in get_numbers():
    dict_words.append(numb)


not_dict_words = [word for word in words if word['lemma']['post'] not in DICT_POST_TYPES]
not_dict_words = [word for word in get_flat_words(not_dict_words)]

dwords_dic = {word.text: word for word in dict_words}

for nd_word in tqdm.tqdm(not_dict_words, desc='Looking for duplicates'):
    if nd_word['text'] in dwords_dic:
        dwords_dic[nd_word['text']].add_gram(nd_word)



root = etree.Element('Tree')
cur_items = [(root, dict_words)]

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
tree.write(open(REZ_PATH, 'wb+'), xml_declaration=True, encoding='utf-8')

logging.info("Tree dictionary released")
logging.info(f"Words count {len(dict_words)}")