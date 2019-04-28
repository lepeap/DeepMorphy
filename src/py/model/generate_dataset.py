import os
import shutil
import random
import pickle
import logging
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize
from utils import get_flat_words, get_grams_info, config

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

    word_vect = np.zeros((MAX_WORD_SIZE,))
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

def save_dataset(items, file_prefix):
    RANDOM.shuffle(items)
    test_size = int(CONFIG['test_persent'] * len(items) / 100)
    valid_size = int(CONFIG['validation_persent'] * len(items) / 100)
    test_items = items[:test_size]
    valid_items = items[-valid_size:]
    items = items[test_size:len(items) - valid_size]

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

    rez_items = []
    for word in tqdm(vec_words, desc=f"Generating classification {cls_type} dataset"):
        y = np.zeros((len(ordered_keys),), dtype=np.int)
        has_classes = False
        for form in vec_words[word]['forms']:
            if cls_type in form:
                index = cls_dic[form[cls_type]]
                y[index] = 1
                has_classes = True

        if has_classes:
            rez_items.append({
                'src': word,
                'x': vec_words[word]['vect'],
                'y': y,
                'weight': weights.reshape(-1, 1)[y == 1].max()
            })

    logging.info(f"Class '{cls_type}': {len(rez_items)}")
    save_dataset(rez_items, cls_type)

def create_lemma_dataset(vec_words, main_cls_dic):
    def process_seq_vect(dic):
        vect = list(dic['vect'][0])
        vect.insert(dic['vect'][1], END_TOKEN)
        vect.insert(0, START_TOKEN)
        vect_len = dic['vect'][1] + 2
        return np.asarray(vect), vect_len

    seq_vecs = {}
    rez_items = []
    for word in tqdm(vec_words, desc="Generating lemma dataset"):
        dic = vec_words[word]
        if word in seq_vecs:
            x_vec = seq_vecs[word]
        else:
            x_vec = process_seq_vect(dic)
            seq_vecs[word] = x_vec

        for form in dic['forms']:
            main_cls = main_cls_dic[form['main']]

            if 'lemma' in form:
                word = form['lemma']
                y_src = word
                if word in seq_vecs:
                    y_vec = seq_vecs[word]
                else:
                    y_vec = process_seq_vect(vec_words[word])
                    seq_vecs[word] = y_vec
            else:
                y_src = word
                y_vec = x_vec

            rez_items.append({
                'x_src': word,
                'x': x_vec[0],
                'x_len': x_vec[1],
                'y_src': y_src,
                'y': y_vec[0],
                'y_len': y_vec[1],
                'main_cls': main_cls
            })

    save_dataset(rez_items, 'lemma')
    logging.info(f"Lemmatizer: {len(rez_items)}")



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

create_datasets(words_dic)
logging.info("Dataset generated")




