import logging
import pickle
from utils import get_grams_info, config
from lxml import etree


CONFIG = config()
DIC_PATH = CONFIG['dict_path']
WORDS_PATH = CONFIG['words_path']
SRC_CONVERT, _ = get_grams_info(CONFIG)

def parse_dic_words(itr):
    cur_word = None
    cur_item = None
    event, element = next(itr)

    while not (event == 'end' and element.tag == 'lemmata'):

        if event == 'start' and element.tag == 'lemma':
            cur_word = {
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
            logging.info("Lemma '%s' extracted from dict", cur_word['lemma']['text'])

            yield cur_word
            cur_word = None

        event, element = next(itr)

doc = etree.iterparse(DIC_PATH, events=('start', 'end'))
itr = iter(doc)
event, element = next(itr)

while not (event == 'start' and element.tag == 'lemmata'):
    event, element = next(itr)

words = list(parse_dic_words(itr))

with open(WORDS_PATH, 'wb+') as f:
    pickle.dump(words, f)