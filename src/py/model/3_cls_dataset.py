import pickle
import logging
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize
from collections import defaultdict
from utils import get_grams_info, CONFIG, save_dataset


VECT_PATH = CONFIG['vect_words_path']
CLS_CLASSES_PATH = CONFIG['cls_classes_path']
GRAMMEMES_TYPES = CONFIG['grammemes_types']
SRC_CONVERT, CLASSES_INDEXES = get_grams_info(CONFIG)


def generate_dataset(vec_words, cls_type, cls_dic):
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


def generate_all(vec_words):
    #for cls_type in CLASSES_INDEXES:
    #    cls_dic = CLASSES_INDEXES[cls_type]
    #    generate_dataset(vec_words, cls_type, cls_dic)

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

    with open(VECT_PATH, 'wb+') as f:
        pickle.dump(vec_words, f)

    cls_dic = {
        tpl: index
        for index, tpl in enumerate(un_classes)
    }
    #generate_dataset(vec_words, 'main', cls_dic)
    #logging.info(f"Main classes count: {len(cls_dic)}")
    with open(CLS_CLASSES_PATH, 'wb+') as f:
        pickle.dump(cls_dic, f)


with open(VECT_PATH, 'rb') as f:
    vwords = pickle.load(f)

generate_all(vwords)
