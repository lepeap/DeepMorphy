import json
import pickle
import numpy as np
from collections import defaultdict
from utils import CONFIG, RANDOM, MAX_WORD_SIZE, get_grams_info, create_cls_tuple, vectorize_text, save_dataset

GEN_MAX_COMBINE_ITEMS = 5
MIN_SPLIT_LENGTH = 8
CHARS = CONFIG['chars']
CHARS_INDEXES = {c: index for index, c in enumerate(CHARS)}
AMB_CHARS = CHARS + CONFIG['ambig_chars']
AMB_CHARS_INDEXES = {c: index for index, c in enumerate(AMB_CHARS)}

SRC_CONVERT, _ = get_grams_info(CONFIG)
for key in CONFIG['dict_post_types']:
    SRC_CONVERT[key] = ('post', key)

for key in CONFIG['other_post_types']:
    SRC_CONVERT[key] = ('post', key)
SRC_CONVERT['pnct'] = ('post', 'punct')
SRC_CONVERT['numb'] = ('post', 'int')

with open(CONFIG['tags_path'], 'rb') as f:
    tags = pickle.load(f)

MASK_DIC = defaultdict(lambda: 1)
MASK_DIC[tags[create_cls_tuple({'post': 'punct'})]['i']] = 0
MASK_DIC[tags[create_cls_tuple({'post': 'unkn'})]['i']] = 0
MASK_DIC[tags[create_cls_tuple({'post': 'int'})]['i']] = 0


def create_item(text, lower_text, tags, tag_id):
    upper_mask = np.zeros((MAX_WORD_SIZE,), np.int)
    for index, c in enumerate(text[:MAX_WORD_SIZE]):
        val = 1 if c.isupper() else 0
        upper_mask[index] = val
    return {
                'text': lower_text,
                'upper_mask': upper_mask,
                'tags': tags,
                'x': vectorize_text(lower_text, CHARS, CHARS_INDEXES),
                'x_amb': vectorize_text(lower_text, AMB_CHARS, AMB_CHARS_INDEXES),
                'y': tag_id,
                'mask': MASK_DIC[tag_id]
            }


def create_dataset(src_items):
    items = []
    for sent in src_items:
        ds_sent = []
        broken = False
        for token in sent:
            grams = [SRC_CONVERT[gram_key.lower()] for gram_key in token['Grams'] if gram_key in SRC_CONVERT]
            gram_dic = {item[0]: item[1] for item in grams}
            cls_tpl = create_cls_tuple(gram_dic)
            if not any([item for item in list(cls_tpl) if item is not None]):
                gram_dic['post'] = 'unkn'
                cls_tpl = create_cls_tuple(gram_dic)

            if cls_tpl not in tags:
                broken = True
                break

            tag_id = tags[cls_tpl]['i']
            text = token['Text'].lower()
            if len(text) > MAX_WORD_SIZE:
                text = text[:MAX_WORD_SIZE]
            ds_sent.append(create_item(token['Text'], text, token['Tags'], tag_id))

        if not broken:
            items.append(ds_sent)
    return items


def select_items(items, count):
    rez_items = []
    while len(rez_items) < count:
        i = RANDOM.randint(0, len(items) - 1)
        rez_items.append(items[i])
        del items[i]
    return rez_items


def generate_combine(src_train):
    simple_items = []
    punct = [',', ':', 'ー']
    for sent in src_train:
        if any([item for item in sent if item['text'] in punct]):
            continue

        simple_items.append(sent)

    rez_count = len(simple_items)
    rez_items = []
    connectors = ['.', '!', '?']
    connector_tag_id = tags[create_cls_tuple({'post': 'punct'})]['i']
    connectors = [create_item(c, c, [connector_tag_id], connector_tag_id) for c in connectors]
    while len(rez_items) != rez_count:
        rez_sent = []
        rez_items.append(rez_sent)
        count = RANDOM.randint(2, GEN_MAX_COMBINE_ITEMS)
        for i in range(count):
            sent = simple_items[RANDOM.randint(0, len(simple_items) - 1)]
            rez_sent.extend(sent)
            if rez_sent[-1]['y'] != connector_tag_id:
                rez_sent.append(connectors[RANDOM.randint(0, len(connectors)-1)])
    return rez_items


def generate_split(src_train):
    res_items = []
    punct = [',', ':', 'ー']
    for sent in src_train:
        spl_indexes = []
        for index, item in enumerate(sent):
            if item['text'] in punct:
                spl_indexes.append(index)

        if len(spl_indexes) == 0:
            continue

        prev_index = 0
        while len(spl_indexes):
            index = spl_indexes[0]
            if index - prev_index >= MIN_SPLIT_LENGTH:
                res_items.append(sent[prev_index:index])
                prev_index = index + 1
            del spl_indexes[0]
    return res_items


def generate_additional(src_train):
    additional_items = generate_split(src_train)
    additional_items += generate_combine(src_train)
    return additional_items


def split_dataset(dataset):
    total = len(dataset)
    train = dataset
    test_count = total * CONFIG['test_persent'] / 100
    valid_count = total * CONFIG['validation_persent'] / 100
    test = select_items(train, test_count)
    valid = select_items(train, valid_count)
    #train += generate_additional(train)
    print("Train count: {0}; Valid count: {1}; Test count: {2}".format(len(train),len(valid), len(test)))
    RANDOM.shuffle(test)
    RANDOM.shuffle(valid)
    RANDOM.shuffle(train)
    save_dataset('ambig', train, valid, test)


src_path = CONFIG['ambig_src_path']
with open(src_path, 'r') as f:
    sents = json.load(f)

ds_items = create_dataset(sents)
split_dataset(ds_items)
