import re
import os
import gzip
import pickle
from utils import CONFIG, get_dict_path, load_datasets


NAR_REG = re.compile("\d+-.*")
RANDOM_SEED = 1917
VECT_PATH = CONFIG['vect_words_path']
DATASET_PATH = CONFIG['dataset_path']
REZ_PATHS = CONFIG['publish_dictionary_paths']
DICT_WORDS_PATH = CONFIG['dict_words_path']
NOT_DICT_WORDS_PATH = CONFIG['dataset_words_path']
MAX_WORD_SIZE = CONFIG['max_word_size']
DICT_POST_TYPES = CONFIG['dict_post_types']
GRAMMEMES_TYPES = CONFIG['grammemes_types']
IGNORE_AD_TAGS = CONFIG['dict_ignore_tags']
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

ad_tags_dict = {}
with open(VECT_PATH, 'rb') as f:
    vec_words = pickle.load(f)
    for word in vec_words:
        item = vec_words[word]
        for form in item['forms']:
            lexeme_id_key = 'inflect_id' if 'inflect_id' in form else 'id'
            lexeme_id = form[lexeme_id_key]

            if 'ad_tags' not in form:
                continue

            if lexeme_id not in ad_tags_dict:
                ad_tags_dict[lexeme_id] = set()

            ad_tags_dict[lexeme_id].add(form['ad_tags'])


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

        order = []
        cur_forms_dict = {}
        for item in words_dics[id]:
            if item['text'] not in cur_forms_dict:
                cur_forms_dict[item['text']] = {}

            replace_other = item['replace_other'] if 'replace_other' in item else False
            cur_form_dic = cur_forms_dict[item['text']]
            if item['main'] not in cur_forms_dict or not cur_forms_dict[item['main']]:
                cur_form_dic[item['main']] = replace_other

            if item['text'] not in order:
                order.append(item['text'])

        for text in order:
            cur_lexeme.append(text)
            cur_lexeme.append(':')
            for cls in cur_forms_dict[text]:
                replace_other = cur_forms_dict[text][cls]
                cur_lexeme.append(str(cls))
                if replace_other:
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

    for id in dict_words:
        un_id_dict = {}
        rez_list = []
        for word in sorted(dict_words[id], key=lambda x: x['index']):
            if word['main'] not in un_id_dict:
                un_id_dict[word['main']] = word
            else:
                un_id_dict[word['main']]['replace_other'] = True
            rez_list.append(word)
        dict_words[id] = rez_list

    index, lexeme = create_dictionary(dict_words)
    save_dictionary(index, lexeme, REZ_PATHS, 'dict')


def release_correction_items():
    dict_words = {}

    with open(get_dict_path('lemma'), 'rb') as f:
        items = pickle.load(f)
    for word in items:
        lexeme_id = word['id']
        if lexeme_id not in lemma_dict \
            or (lexeme_id in ad_tags_dict and any([key in ad_tags_dict[lexeme_id] for key in IGNORE_AD_TAGS])):
            continue

        if lexeme_id not in dict_words:
            dict_words[lexeme_id] = []

        dict_words[lexeme_id].append(word)
        dict_words[lexeme_id].append({
            'id': lexeme_id,
            'text': lemma_dict[lexeme_id][0],
            'main': lemma_dict[lexeme_id][1],
        })

    with open(os.path.join(CONFIG['bad_path'], "bad_lemma.pkl"), 'rb') as f:
        items = pickle.load(f)
    for word in items:
        word = word[0]
        lexeme_id = word['id']
        if lexeme_id not in lemma_dict \
            or (lexeme_id in ad_tags_dict and any([key in ad_tags_dict[lexeme_id] for key in IGNORE_AD_TAGS])):
            continue

        if lexeme_id not in dict_words:
            dict_words[lexeme_id] = []

        lemma, lemma_cls = lemma_dict[lexeme_id]
        dict_words[lexeme_id].append(dict(id=lexeme_id, main=word['main_cls'], text=word['x_src'], replace_other=True))
        dict_words[lexeme_id].append(dict(id=lexeme_id, main=lemma_cls, text=lemma, replace_other=True))

    with open(os.path.join(CONFIG['bad_path'], "bad_inflect.pkl"), 'rb') as f:
        items = pickle.load(f)
    for word in items:
        word = word[0]
        lexeme_id = word['id']
        if lexeme_id not in lemma_dict \
            or (lexeme_id in ad_tags_dict and any([key in ad_tags_dict[lexeme_id] for key in IGNORE_AD_TAGS])):
            continue

        if lexeme_id not in dict_words:
            dict_words[lexeme_id] = []
        dict_words[lexeme_id].append(dict(id=lexeme_id, main=word['x_cls'], text=word['x_src'], replace_other=True))
        dict_words[lexeme_id].append(dict(id=lexeme_id, main=word['y_cls'], text=word['y_src'], replace_other=True))

    with open(os.path.join(CONFIG['bad_path'], "bad_main.pkl"), 'rb') as f:
        items = pickle.load(f)
    for bad_item in items:
        text = bad_item[0]['src']
        for word in vec_words[text]['forms']:
            lexeme_id = word['id']
            if lexeme_id not in lemma_dict \
                or (lexeme_id in ad_tags_dict and any([key in ad_tags_dict[lexeme_id] for key in IGNORE_AD_TAGS])):
                continue

            if lexeme_id not in dict_words:
                dict_words[lexeme_id] = []

            cls_id = tpl_cls_dict[word['main']]['i']
            dict_words[lexeme_id].append(dict(id=lexeme_id, main=cls_id, text=text, replace_other=True))
            dict_words[lexeme_id].append(dict(id=lexeme_id, main=lemma_dict[lexeme_id][1], text=lemma_dict[lexeme_id][0], replace_other=True))

    index, lexeme = create_dictionary(dict_words)
    save_dictionary(index, lexeme, REZ_PATHS, 'dict_correction')


release_correction_items()
release_dict_items()

