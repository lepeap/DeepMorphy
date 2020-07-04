import os
import shutil
import random
import pickle
import logging
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize
from collections import defaultdict
from utils import get_grams_info, config

CONFIG = config()
RANDOM_SEED = 1917
RANDOM = random.Random(RANDOM_SEED)
DATASET_PATH = CONFIG['dataset_path']
WORDS_PATH = CONFIG['dataset_words_path']
MAX_WORD_SIZE = CONFIG['max_word_size']
GRAMMEMES_TYPES = CONFIG['grammemes_types']
CHARS = CONFIG['chars']
SRC_CONVERT, CLASSES_INDEXES = get_grams_info(CONFIG)
CHARS_INDEXES = {c: index for index, c in enumerate(CHARS)}
START_TOKEN = CONFIG['start_token']
END_TOKEN = CONFIG['end_token']



def vectorize_text(text):
    word_vect = np.full((MAX_WORD_SIZE,), END_TOKEN, dtype=np.int32)
    for index, c in enumerate(text):
        if c in CHARS:
            word_vect[index] = CHARS_INDEXES[c]
        else:
            word_vect[index] = CHARS_INDEXES["UNDEFINED"]

    seq_len = len(text)
    return word_vect, seq_len


def vectorize_words(words_dic):
    vect_dic = {}
    for word in tqdm(words_dic, desc="Vectorizing words"):
        vect_dic[word] = {
            'vect': vectorize_text(word),
            'forms': words_dic[word]
        }
        if 'ё' in word:
            forms = []
            for form in words_dic[word]:
                form = dict(form)
                form['text'] = form['text'].replace('ё', 'е')
                if 'lemma' in form:
                    form['lemma'] = form['lemma'].replace('ё', 'е')
                forms.append(form)
            word = word.replace('ё', 'е')
            vect_dic[word] = {
                'vect': vectorize_text(word),
                'forms': forms
            }

    return vect_dic


def select_uniform_items(items_dict, persent, ds_info):
    for cls in tqdm(items_dict, desc=f"Selecting {ds_info} dataset"):
        i = 0
        items = items_dict[cls]
        per_group_count = persent * len(items) / 100

        while i <= per_group_count and len(items) > 0:
            item = items[0]
            items.remove(item)
            yield item
            i += 1


def save_dataset(items_dict, file_prefix):
    total_count = sum([len(items_dict[key]) for key in items_dict])
    logging.info(f"Class '{file_prefix}': {total_count}")
    for key in tqdm(items_dict, desc=f"Shuffling {file_prefix} items"):
        RANDOM.shuffle(items_dict[key])

    test_items = list(select_uniform_items(items_dict, CONFIG['test_persent'], f"test {file_prefix}"))
    valid_items = list(select_uniform_items(items_dict, CONFIG['validation_persent'], f"valid {file_prefix}"))
    items = []
    for key in items_dict:
        items.extend(items_dict[key])
    RANDOM.shuffle(items)

    logging.info(f"Saving '{file_prefix}' train dataset")
    with open(os.path.join(DATASET_PATH, f"{file_prefix}_train_dataset.pkl"), 'wb+') as f:
        pickle.dump(items, f)

    logging.info(f"Saving '{file_prefix}' valid dataset")
    with open(os.path.join(DATASET_PATH, f"{file_prefix}_valid_dataset.pkl"), 'wb+') as f:
        pickle.dump(valid_items, f)

    logging.info(f"Saving '{file_prefix}' test dataset")
    with open(os.path.join(DATASET_PATH, f"{file_prefix}_test_dataset.pkl"), 'wb+') as f:
        pickle.dump(test_items, f)


def generate_classification_dataset(vec_words, cls_type, cls_dic):
    ordered_keys = [cls for cls in sorted(cls_dic, key=lambda cls: cls_dic[cls])]

    weights = [0 for key in ordered_keys]
    for word in tqdm(vec_words, desc=f"Calculating {cls_type} weights"):
        for form in vec_words[word]['forms']:
            if cls_type in form:
                i = cls_dic[form[cls_type]]
                weights[i] = weights[i] + 1

    weights = normalize(np.asarray(weights).reshape(1, -1))
    weights = np.ones((len(ordered_keys),)) - weights

    rez_items = defaultdict(list)
    cur_cls = None
    for word in tqdm(vec_words, desc=f"Generating classification {cls_type} dataset"):
        y = np.zeros((len(ordered_keys),), dtype=np.int)
        has_classes = False
        for form in vec_words[word]['forms']:
            if cls_type in form:
                cur_cls = form[cls_type]
                index = cls_dic[cur_cls]
                y[index] = 1
                has_classes = True

        if has_classes:
            items = rez_items[cur_cls]
            items.append({
                'src': word,
                'x': vec_words[word]['vect'],
                'y': y,
                'weight': weights.reshape(-1, 1)[y == 1].max()
            })
            rez_items[cur_cls] = items

    save_dataset(rez_items, cls_type)


def create_lemma_dataset(vec_words, main_cls_dic):
    rez_dict = defaultdict(list)
    for word in tqdm(vec_words, desc="Generating lemma dataset"):
        dic = vec_words[word]
        x_vec = dic['vect']

        for form in dic['forms']:
            main_cls = main_cls_dic[form['main']]

            if 'lemma' in form:
                word_y = form['lemma']
            else:
                continue

            y_vec = vec_words[word_y]['vect']

            items = rez_dict[main_cls]
            items.append({
                'x_src': word,
                'x': x_vec[0],
                'x_len': x_vec[1],
                'y_src': word_y,
                'y': y_vec[0],
                'y_len': y_vec[1],
                'main_cls': main_cls
            })
            rez_dict[main_cls] = items

    save_dataset(rez_dict, 'lemma')



def create_datasets(words):
    if os.path.isdir(DATASET_PATH):
        shutil.rmtree(DATASET_PATH)

    os.mkdir(DATASET_PATH)
    vec_words = vectorize_words(words)
    for cls_type in CLASSES_INDEXES:
        cls_dic = CLASSES_INDEXES[cls_type]
        generate_classification_dataset(vec_words, cls_type, cls_dic)

    un_classes = []
    for word in tqdm(vec_words, desc="Setting main class"):
        for form in vec_words[word]['forms']:
            tpl = tuple(
                form[key] if key in form else None
                for key in GRAMMEMES_TYPES
            )
            if tpl not in un_classes:
                un_classes.append(tpl)
            form['main'] = tpl

    cls_dic = {
        tpl: index
        for index, tpl in enumerate(un_classes)
    }
    generate_classification_dataset(vec_words, 'main', cls_dic)
    logging.info(f"Main classes count: {len(cls_dic)}")
    with open(os.path.join(DATASET_PATH, f"classification_classes.pkl"), 'wb+') as f:
        pickle.dump(cls_dic, f)

    create_lemma_dataset(vec_words, cls_dic)




with open(WORDS_PATH, 'rb') as f:
    words_dic = pickle.load(f)

#rez_dict = {}
#i = 0
#for key in words_dic:
#    rez_dict[key] = words_dic[key]
#    i += 1
#    if i > 100000:
#        break
#words_dic = rez_dict

create_datasets(words_dic)
logging.info("Dataset generated")




