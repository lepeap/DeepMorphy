import string
import re
import os
import gzip
import tqdm
import pickle
import logging
from utils import CONFIG, get_dict_path, load_datasets


NAR_REG = re.compile("\d+-.*")
RANDOM_SEED = 1917
DATASET_PATH = CONFIG['dataset_path']
REZ_PATHS = CONFIG['publish_dictionary_paths']
DICT_WORDS_PATH = CONFIG['dict_words_path']
NOT_DICT_WORDS_PATH = CONFIG['dataset_words_path']
MAX_WORD_SIZE = CONFIG['max_word_size']
DICT_POST_TYPES = CONFIG['dict_post_types']
GRAMMEMES_TYPES = CONFIG['grammemes_types']

REPLACE_WORD_DICT_ID = 1

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

with open(CONFIG['inflect_templates_path'], 'rb') as f:
    inflect_templates = pickle.load(f)

with open(CONFIG['tags_path'], 'rb') as f:
    tpl_cls_dict = pickle.load(f)

lemma_cls_dict = {}
for lemma_tpl in inflect_templates:
    lemma_id = tpl_cls_dict[lemma_tpl]['i']
    for tpl in inflect_templates[lemma_tpl]:
        lemma_cls_dict[tpl_cls_dict[tpl]['i']] = lemma_id

lemma_dict = {}
for item in load_datasets('inflect', 'test', 'train', 'valid'):
    if item['id'] not in lemma_dict:
        lemma_dict[item['id']] = (item['x_src'], item['x_cls'])


def build_index(words_dics):
    text_forms_dict = {}
    for id in words_dics:
        for item in words_dics[id]:
            text = item['text']
            if text not in text_forms_dict:
                text_forms_dict[text] = []
            text_forms_dict[text].append(item)

    index = []
    for text in text_forms_dict:
        lexemes = [str(item['id']) for item in text_forms_dict[text]]
        lexemes = ','.join(lexemes)
        index.append(f"{text}:{lexemes}")

    index = list(set(index))
    index = '\n'.join(index)
    return index


def create_dictionary(words_dics):
    index = build_index(words_dics)

    lexeme = []
    for id in words_dics:
        cur_lexeme = [id, '\t']

        cur_forms_dict = {}
        for item in words_dics[id]:
            if item['text'] not in cur_forms_dict:
                cur_forms_dict[item['text']] = []
            reaplace_other = item['reaplace_other'] if 'reaplace_other' in item else False
            cur_forms_dict[item['text']].append((item['main'], reaplace_other))

        for text in cur_forms_dict:
            cur_lexeme.append(text)
            cur_lexeme.append(':')
            for cls, reaplace_other in cur_forms_dict[text]:
                cur_lexeme.append(str(cls))
                if reaplace_other is not None:
                    cur_lexeme.append('!')

                cur_lexeme.append(",")

            del cur_lexeme[-1]
            cur_lexeme.append(';')

        del cur_lexeme[-1]
        lexeme.append(''.join(cur_lexeme))

    lexeme = '\n'.join(lexeme)
    return index, lexeme


def save_dictionary(index, lexeme, paths, file_name):
    for path in paths:
       path = os.path.join(path, f"{file_name}_index.txt.gz")
       with gzip.open(path, 'wb+') as f:
           f.write(index.encode('utf-8'))

    for path in paths:
       path = os.path.join(path, f"{file_name}.txt.gz")
       with gzip.open(path, 'wb+') as f:
           f.write(lexeme.encode('utf-8'))


def release_dict_items():
    with open(DICT_WORDS_PATH, 'rb') as f:
        words = pickle.load(f)

    words = [word for word in words if word['post'] != 'numb']
    dict_words = {}
    for word in words:
        if word['id'] not in dict_words:
            dict_words[word['id']] = []
        dict_words[word['id']].append(word)

    index, lexeme = create_dictionary(dict_words)
    save_dictionary(index, lexeme, REZ_PATHS, 'dict')


def release_correction_items():
    dict_words = {}
    with open(get_dict_path('lemma'), 'rb') as f:
        items = pickle.load(f)
    for word in items:
        if word['id'] not in lemma_dict:
            continue

        if word['id'] not in dict_words:
            dict_words[word['id']] = []

        dict_words[word['id']].append(word)
        dict_words[word['id']].append({
            'id': word['id'],
            'text': lemma_dict[word['id']][0],
            'main': lemma_dict[word['id']][1],
        })

    with open(os.path.join(CONFIG['bad_path'], "bad_lemma.pkl"), 'rb') as f:
        items = pickle.load(f)
    for word in items:
        word = word[0]
        if word['id'] not in lemma_dict:
            continue

        if word['id'] not in dict_words:
            dict_words[word['id']] = []

        lemma, lemma_cls = lemma_dict[word['id']]
        dict_words[word['id']].append(dict(id=word['id'], main=word['main_cls'], text=word['x_src'], replace_other=True))
        dict_words[word['id']].append(dict(id=word['id'], main=lemma_cls, text=lemma, replace_other=True))

    with open(os.path.join(CONFIG['bad_path'], "bad_inflect.pkl"), 'rb') as f:
        items = pickle.load(f)
    for word in items:
        word = word[0]
        if word['id'] not in lemma_dict:
            continue

        if word['id'] not in dict_words:
            dict_words[word['id']] = []
        dict_words[word['id']].append(dict(id=word['id'], main=word['x_cls'], text=word['x_src'], replace_other=True))
        dict_words[word['id']].append(dict(id=word['id'], main=word['y_cls'], text=word['y_src'], replace_other=True))

    index, lexeme = create_dictionary(dict_words)
    save_dictionary(index, lexeme, REZ_PATHS, 'dict_correction')


release_correction_items()
release_dict_items()
print()


#with open(NOT_DICT_WORDS_PATH, 'rb') as f:
#    not_dict_words = pickle.load(f)
#    io_words = [io for io in not_dict_words if 'ё' in io]
#    for io_word in io_words:
#        io_items = not_dict_words[io_word]
#        not_io_word = io_word.replace('ё', 'е')
#        not_io_items = []
#        for item in io_items:
#            not_io_item = dict(item)
#            if 'lemma' in not_io_item:
#                not_io_item['lemma'] = not_io_item['lemma'].replace('ё', 'е')
#            not_io_item['text'] = not_io_item['text'].replace('ё', 'е')
#            not_io_items.append(not_io_item)
#
#        not_dict_words[not_io_word] = not_io_items
#
#if with_wrongs:
#    with open("wrong_words.pkl", 'rb') as f:
#        wrongs = set(pickle.load(f))
#else:
#    wrongs = set()
#
#dwords_dic = {word.text: word for word in words}
#for nd_word in tqdm.tqdm(not_dict_words, desc='Looking for duplicates'):
#    if nd_word in dwords_dic and not NAR_REG.match(nd_word):
#        for gram in not_dict_words[nd_word]:
#            dwords_dic[nd_word].add_gram(gram)
#    elif nd_word in wrongs:
#        items = not_dict_words[nd_word]
#        word = Word(items[0])
#        del items[0]
#        for item in items:
#            word.add_gram(item)
#
#        words.append(word)
#
#ind_keys = string.ascii_lowercase + string.ascii_uppercase
#index_dict = {}
#cur_index = 0
#txt = []
#for word in words:
#    txt.append(word.text)
#    txt.append('\t')
#    for ind, g in enumerate(word.get_grams()):
#        if 'lemma' in g and g['lemma']:
#            txt.append(g['lemma'])
#        txt.append(':')
#        for g_ind, val in enumerate(g['gram'].split(',')):
#            if val and val not in index_dict:
#                index_dict[val] = ind_keys[cur_index]
#                cur_index+=1
#
#            if val:
#                txt.append(index_dict[val])
#
#            if g_ind != len(GRAMMEMES_TYPES):
#                txt.append(',')
#
#        if ind != len(word.grams)-1:
#            txt.append(';')
#
#    txt.append('\n')
#
#txt.insert(0, "\n")
#for key in index_dict:
#    txt.insert(0, f"{index_dict[key]}={key}\n")
#
#txt = "".join(txt)
#rez = txt.encode('utf-8')
#
#for path in REZ_PATHS:
#    path = os.path.join(path, "dict.txt.gz")
#    with gzip.open(path, 'wb+') as f:
#        f.write(rez)
#
#logging.info("Dictionary released")
#logging.info(f"Words count {len(words)}")
