import pickle
from utils import CONFIG, create_cls_tuple

gram_types = CONFIG['grammemes_types']
with open(CONFIG['numb_classes_path'], 'rb') as f:
    numb_classes_dic = pickle.load(f)
with open(CONFIG['numb_data_path'], 'rb') as f:
    numb_data = pickle.load(f)

lemma_same_words = []
for cls in CONFIG['lemma_same_word']:
    lemma_same_words.append(create_cls_tuple(cls))

items = {}
classes_dic = CONFIG['main_classes']
for tpl in classes_dic:
    items[tpl] = {
        'i': classes_dic[tpl],
        'p': 'nn',
        'l':  tpl in lemma_same_words
    }

for tpl in numb_classes_dic:
    cls_index = numb_classes_dic[tpl]
    items[tpl] = {
        'i': cls_index,
        'p': 'numb',
        'l': cls_index in numb_data['lemma_cls_ids']
    }

max_cls_id = max([numb_classes_dic[key] for key in numb_classes_dic])
max_cls_id += 1
post_index = gram_types['post']['index']
for key in CONFIG['other_post_types']:
    tpl = [None for item in CONFIG['grammemes_types']]
    tpl[post_index] = key
    tpl = tuple(tpl)
    items[tpl] = {
        'i': max_cls_id,
        'p': 'reg',
        'l':  max_cls_id
    }
    max_cls_id += 1

tpl = [None for item in CONFIG['grammemes_types']]
tpl[post_index] = 'unkn'
tpl = tuple(tpl)
items[tpl] = {
    'i': max_cls_id,
    'p': 'reg',
    'l':  max_cls_id
}
max_cls_id += 1

with open(CONFIG['dict_words_path'], 'rb') as f:
    dic_words = pickle.load(f)

for dic_item in dic_words:
    cls_tpl = create_cls_tuple(dic_item)
    if cls_tpl not in items:
        items[cls_tpl] = {
            'i': max_cls_id,
            'p': 'dict',
            'l': ('npro' in cls_tpl and 'nomn' in cls_tpl) or cls_tpl in lemma_same_words
        }
        max_cls_id += 1

    dic_item['main'] = items[cls_tpl]['i']

with open(CONFIG['dict_words_path'], 'wb+') as f:
    pickle.dump(dic_words, f)


npos = [item for item in items if 'npro' in item]
with open(CONFIG['tags_path'], 'wb+') as f:
    pickle.dump(items, f)
