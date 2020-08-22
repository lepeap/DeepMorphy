import os
import yaml
import random
import pickle
import logging
from collections import defaultdict
from tqdm import tqdm


def _get_config():
    with open('config.yml', 'r') as f:
        config = yaml.load(f)

    classes_path = config['cls_classes_path']
    if os.path.exists(classes_path):
        with open(classes_path, 'rb') as f:
            config['main_classes'] = pickle.load(f)
            config['main_classes_count'] = len(config['main_classes'])

    return config


CONFIG = _get_config()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S'
)
DATASET_PATH = CONFIG['dataset_path']
DICS_PATH = CONFIG['dics_path']
RANDOM = random.Random(CONFIG['random_seed'])
GRAMMEMES_TYPES = CONFIG['grammemes_types']


class MyDefaultDict(defaultdict):
    def __missing__(self, key):

        if self.default_factory is None:
            raise KeyError( key )
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def get_grams_info(config):
    dict_post_types = config['dict_post_types']
    grammemes_types = config['grammemes_types']
    src_convert = {}
    classes_indexes = {}

    for gram_key in grammemes_types:
        gram = grammemes_types[gram_key]
        cls_dic = gram['classes']
        classes_indexes[gram_key] = {}
        for cls_key in cls_dic:
            cls_obj = cls_dic[cls_key]
            classes_indexes[gram_key][cls_key] = cls_obj['index']

            for key in cls_obj['keys']:
                src_convert[key.lower()] = (gram_key, cls_key)

    for post_key in dict_post_types:
        cls_obj = dict_post_types[post_key.lower()]
        p_key = ('post', post_key.lower())
        for key in cls_obj['keys']:
            src_convert[key.lower()] = p_key

    return src_convert, classes_indexes


def decode_word(vect_mas):
    conf = CONFIG
    word = []
    for ci in vect_mas:
        if ci == conf['end_token']:
            return "".join(word)
        elif ci < len((conf['chars'])):
            word.append(conf['chars'][ci])
        else:
            word.append("0")

    return "".join(word)


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
    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    total_count = sum([len(items_dict[key]) for key in items_dict])
    logging.info(f"Class '{file_prefix}': {total_count}")
    for key in tqdm(items_dict, desc=f"Shuffling {file_prefix} items"):
        random.shuffle(items_dict[key])

    test_items = list(select_uniform_items(items_dict, CONFIG['test_persent'], f"test {file_prefix}"))
    valid_items = list(select_uniform_items(items_dict, CONFIG['validation_persent'], f"valid {file_prefix}"))
    items = []
    for key in items_dict:
        items.extend(items_dict[key])
    random.shuffle(items)

    logging.info(f"Saving '{file_prefix}' train dataset")
    with open(os.path.join(DATASET_PATH, f"{file_prefix}_train_dataset.pkl"), 'wb+') as f:
        pickle.dump(items, f)

    logging.info(f"Saving '{file_prefix}' valid dataset")
    with open(os.path.join(DATASET_PATH, f"{file_prefix}_valid_dataset.pkl"), 'wb+') as f:
        pickle.dump(valid_items, f)

    logging.info(f"Saving '{file_prefix}' test dataset")
    with open(os.path.join(DATASET_PATH, f"{file_prefix}_test_dataset.pkl"), 'wb+') as f:
        pickle.dump(test_items, f)


def get_dict_path(file_prefix):
    return os.path.join(DICS_PATH, f"{file_prefix}_dict_items.pkl")


def save_dictionary_items(items, file_prefix):
    with open(get_dict_path(file_prefix), 'wb+') as f:
        pickle.dump(items, f)


def create_cls_tuple(item):
    return tuple(
        item[key] if key in item else None
        for key in GRAMMEMES_TYPES
    )

def load_datasets(main_type, *ds_type):
    words = []

    def load_words(type):
        path = os.path.join(CONFIG['dataset_path'], f"{main_type}_{type}_dataset.pkl")
        with open(path, 'rb') as f:
            words.extend(pickle.load(f))

    for key in ds_type:
        load_words(key)

    return words