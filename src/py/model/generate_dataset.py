import os
import shutil
import random
import pickle
import logging
import numpy as np
from itertools import groupby
from utils import get_flat_words, get_grams_info, config

CONFIG = config()
RANDOM_SEED = 1917
RANDOM = random.Random(RANDOM_SEED)
DATASET_PATH = CONFIG['dataset_path']
WORDS_PATH = CONFIG['words_path']
MAX_WORD_SIZE = CONFIG['max_word_size']
GRAMMEMES_TYPES = CONFIG['grammemes_types']
CHARS = CONFIG['chars']
SRC_CONVERT, CLASSES_INDEXES = get_grams_info(CONFIG)
CHARS_INDEXES = {c: index for index, c in enumerate(CHARS)}
DICT_POST_TYPES = CONFIG['dict_post_types']

def vectorize_words_for_classification(words):
    for word in get_flat_words(words):
        item = vectorize_word(word)
        if item is not None:
            logging.info("Word '%s' vectorized", word['text'])
            yield word, item

        if 'ё' in word['text']:
            logging.info("Ё replacement in '%s'", word['text'])
            word = dict(word)
            word['text'] = word['text'].replace('ё', 'е')
            item = vectorize_word(word)
            if item is not None:
                logging.info("Word '%s' vectorized", word['text'])
                yield word, item


def vectorize_word(word):
    if len(word['text']) > MAX_WORD_SIZE:
        logging.warning(f"Ignore word '%s'. Too long.", word['text'])
        return None

    word_vect = np.zeros((MAX_WORD_SIZE,))
    for index, c in enumerate(word['text']):
        if c in CHARS:
            word_vect[index] = CHARS_INDEXES[c]
        else:
            word_vect[index] = CHARS_INDEXES["UNDEFINED"]

    seq_len = len(word['text'])

    return word_vect, seq_len


def create_classifications_datasets(words):
    if os.path.isdir(DATASET_PATH):
        shutil.rmtree(DATASET_PATH)

    os.mkdir(DATASET_PATH)
    vec_words = list(vectorize_words_for_classification(words))
    for cls_type in CLASSES_INDEXES:
        cls_dic = CLASSES_INDEXES[cls_type]
        cls_weght = {
            cls: sum(1 if item[0][cls_type] == cls else 0 for item in vec_words if cls_type in item[0])
            for cls in cls_dic
        }
        max_weight = max(cls_weght[key] for key in cls_weght)

        cls_weght = {
            key: 1.01 - (cls_weght[key] / max_weight)
            for key in cls_weght
        }
        items = [
            (item[1], cls_dic[item[0][cls_type]], cls_weght[item[0][cls_type]])
            for item in vec_words
            if cls_type in item[0]
        ]
        logging.info(f"Class '{cls_type}': {len(items)}")
        create_classification_dataset(items, cls_type)


    rez_cls = [
        tuple(
            tpl[0][key] if key in tpl[0] else None
            for key in GRAMMEMES_TYPES
        )
        for tpl in vec_words
    ]
    main_classes_dic = {
        tpl: index
        for index, tpl in enumerate(list(set(rez_cls)))
    }
    main_classes_weight = {
        key: sum(1 for item in group)
        for key, group in groupby(rez_cls)
    }
    max_main_class_weight = max(main_classes_weight[key] for key in main_classes_weight)
    main_classes_weight = {
        key: 1.01 - (main_classes_weight[key] / max_main_class_weight)
        for key in main_classes_weight
    }
    word_main_cls = [
        main_classes_dic[tpl]
        for tpl in rez_cls
    ]
    word_main_wight = [
        main_classes_weight[tpl]
        for tpl in rez_cls
    ]
    items = list(
        zip(
            [item[1] for item in vec_words],
            word_main_cls,
            word_main_wight
        )
    )
    logging.info(f"Classification classes count: {len(main_classes_dic)}")
    logging.info(f"Classification: {len(items)}")
    create_classification_dataset(items, "classification")

    with open(os.path.join(DATASET_PATH, f"classification_classes.pkl"), 'wb+') as f:
        pickle.dump(main_classes_dic, f)


def create_classification_dataset(items, file_prefix):
    RANDOM.shuffle(items)
    test_size = int(CONFIG['test_persent'] * len(items) / 100)
    valid_size = int(CONFIG['validation_persent'] * len(items) / 100)
    test_items = items[:test_size]
    valid_items = items[-valid_size:]
    items = items[test_size:len(items) - valid_size]

    with open(os.path.join(DATASET_PATH, f"{file_prefix}_train_dataset.pkl"), 'wb+') as f:
        pickle.dump(items, f)

    with open(os.path.join(DATASET_PATH, f"{file_prefix}_valid_dataset.pkl"), 'wb+') as f:
        pickle.dump(valid_items, f)

    with open(os.path.join(DATASET_PATH, f"{file_prefix}_test_dataset.pkl"), 'wb+') as f:
        pickle.dump(test_items, f)

with open(WORDS_PATH, 'rb') as f:
    words = pickle.load(f)

cls_words = [word for word in words if word['lemma']['post'] not in DICT_POST_TYPES]
create_classifications_datasets(cls_words)

logging.info("Dataset generated")




