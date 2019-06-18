import logging
import pickle
from tqdm import tqdm
from utils import get_grams_info, config
from utils import get_flat_words
from lxml import etree


CONFIG = config()
DIC_PATH = CONFIG['dict_path']
MAX_WORD_SIZE = CONFIG['max_word_size']
DATASET_WORDS_PATH = CONFIG['dataset_words_path']
DICTS_WORDS_PATH = CONFIG['dict_words_path']
DICT_POST_TYPES = CONFIG['dict_post_types']
LEMMAS_PROPS = CONFIG['lemma_same_word']
SRC_CONVERT, _ = get_grams_info(CONFIG)
i = 0


def parse_words(itr):
    global i

    cur_word = None
    cur_item = None
    event, element = next(itr)

    while not (event == 'end' and element.tag == 'lemmata'):

        if event == 'start' and element.tag == 'lemma':
            cur_word = {
                'id': element.attrib['id'],
                'lemma': None,
                'forms': []
            }

        if event == 'start' and (element.tag == 'l' or element.tag == 'f'):
            cur_item = {'text': None}

        if event == 'end' and element.tag == 'l':
            cur_item['text'] = element.attrib['t']
            cur_word['lemma'] = cur_item
            cur_item = None

        if event == 'end' and element.tag == 'g' and element.attrib['v'].lower() in SRC_CONVERT:
            src_key = element.attrib['v'].lower()
            gram_type, gram = SRC_CONVERT[src_key]
            cur_item[gram_type] = gram

        if event == 'end' and element.tag == 'f':
            cur_item['text'] = element.attrib['t']
            cur_word['forms'].append(cur_item)
            cur_item = None

        if event == 'end' and element.tag == 'lemma':
            yield cur_word
            cur_word = None

        event, element = next(itr)

def parse_link_types(itr):
    event, element = next(itr)
    link_types = {}

    while not (event == 'start' and element.tag == 'link_types'):
        event, element = next(itr)

    while not (event == 'end' and element.tag == 'link_types'):

        if event == 'end' and element.tag == 'type':
            link_types[element.text] = element.attrib['id']

        event, element = next(itr)


    return link_types


def parse_links(itr):
    event, element = next(itr)

    while not (event == 'end' and element.tag == 'links'):
        if event == 'end' and element.tag == 'link':
            yield {
                'from': element.attrib['from'],
                'to': element.attrib['to'],
                'type': element.attrib['type']
            }
        event, element = next(itr)


def set_lemmas(words, link_types, links):

    lemmas_dict = {}
    for word in words:
        for norm_f in LEMMAS_PROPS:
            is_lemma = True
            for key in norm_f:
                if word[key] != norm_f[key]:
                    is_lemma = False
                    break

            if is_lemma:
                lemmas_dict[word['id']] = word['text']
                del word['lemma']
                break

    inv_link_type_dict = {
        link_types[key]: key
        for key in link_types
    }

    links = {
        (link['to'], inv_link_type_dict[link['type']]) : link['from']
        for link in links
    }

    for word in words:
        link_type = None
        if word['post'] == 'verb':
            link_type = 'INFN-VERB'
        elif word['post'] == 'prtf':
            link_type = 'INFN-PRTF'
        elif word['post'] == 'grnd':
            link_type = 'INFN-GRND'
        elif word['post'] == 'adjs':
            link_type = 'ADJF-ADJS'
        elif word['post'] == 'comp':
            link_type = 'ADJF-COMP'

        if link_type:
            key = (word['id'], link_type)
            if key in links and links[key] in lemmas_dict:
                lemma = lemmas_dict[links[key]]
                word['lemma'] = lemma
                lemmas_dict[word['id']] = lemma
            else:
                del word['lemma']

    for word in words:
        if word['post'] == 'prts':
            key = (word['id'], 'PRTF-PRTS')
            if key in links:
                lemma = lemmas_dict[links[key]]
                word['lemma'] = lemma
                lemmas_dict[word['id']] = lemma
            else:
                del word['lemma']




doc = etree.iterparse(DIC_PATH, events=('start', 'end'))
itr = iter(doc)
event, element = next(itr)
logging.info("Parsing dictionary xml")
while not (event == 'start' and element.tag == 'lemmata'):
    event, element = next(itr)

words = list(parse_words(itr))
link_types = parse_link_types(itr)
links = list(parse_links(itr))
words = list(get_flat_words(words))

set_lemmas(words, link_types, links)

words = [dict(t) for t in {tuple(sorted(d.items())) for d in words}]
dict_words = [word for word in words if word['post'] in DICT_POST_TYPES]
dataset_words = [word for word in words if word['post'] not in DICT_POST_TYPES]

dataset_words_dic = {}
for word in tqdm(dataset_words):
    if len(word['text']) > MAX_WORD_SIZE:
        continue

    if word['text'] not in dataset_words_dic:
        dataset_words_dic[word['text']] = []
    dataset_words_dic[word['text']].append(word)

logging.info(f"Dict words: {len(dict_words)}")
logging.info(f"Dataset words: {len(dataset_words_dic)}")

with open(DATASET_WORDS_PATH, 'wb+') as f:
    pickle.dump(dataset_words_dic, f)

with open(DICTS_WORDS_PATH, 'wb+') as f:
    pickle.dump(dict_words, f)
