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
SRC_CONVERT, _ = get_grams_info(CONFIG)
i = 0

def parse_dic_words(itr):
    global i

    cur_sent = None
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
            yield cur_word
            cur_word = None

            #i += 1
            #if i == 1000:
            #    break

        event, element = next(itr)

def parse_sentences(itr):
    while not (event == 'end' and element.tag == 'paragraphs'):
        if event == 'start' and element.tag == 'lemma':
            cur_word = {
                'lemma': None,
                'forms': []
            }


doc = etree.iterparse(DIC_PATH, events=('start', 'end'))
itr = iter(doc)
event, element = next(itr)


logging.info("Parsing xml")
while not (event == 'start' and element.tag == 'lemmata'):
    event, element = next(itr)

words = list(parse_dic_words(itr))
words = list(get_flat_words(words))
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