import yaml
import pickle
from utils import CONFIG


VECT_PATH = CONFIG['vect_words_path']
CLS_CLASSES_PATH = CONFIG['cls_classes_path']
NMB_CLASSES_PATH = CONFIG['numb_classes_path']
NMB_DATA_PATH =  CONFIG['numb_data_path']
GRAMMEMES_TYPES = CONFIG['grammemes_types']
SOGL_CHARS = ['б', 'в', 'г', 'д', 'ж', 'з', 'й', 'к', 'л', 'м', 'н', 'п', 'р', 'с', 'т', 'ф', 'х', 'ц', 'ч', 'ш', 'щ']
GLASN_CHARS = ['а', 'о', 'и', 'е', 'ё', 'э', 'ы', 'у', 'ю', 'я']


with open(CLS_CLASSES_PATH, 'rb') as f:
    cur_classes_count = len(pickle.load(f)) + 1

with open('numb.yml') as f:
    numbr_src_dic = yaml.load(f)


def get_nar_end(text):
    end = text[-1]
    if text[-2] in SOGL_CHARS and text[-1] in GLASN_CHARS:
        end = text[-2:]

    return end

lemma_cls_id = None
res_dict = {}
numb_cls_dict = {}
for n_key in numbr_src_dic:
    n_key_data = {
        'p': [],
        'o': [],
        'nar_end': {}
    }
    res_dict[n_key] = n_key_data
    for t in numbr_src_dic[n_key]:
        lemma_text = None
        lemma_number_text = None

        for index, item in enumerate(numbr_src_dic[n_key][t]):
            item['post'] = 'numb'
            cls_tpl = tuple(
                item[key] if key in item else None
                for key in GRAMMEMES_TYPES
            )

            if cls_tpl not in numb_cls_dict:
                numb_cls_dict[cls_tpl] = cur_classes_count
                cur_classes_count += 1

            cur_class = numb_cls_dict[cls_tpl]
            if index == 0:
                n_key_data['lemma'] = item['text']

            if index == 0 and lemma_cls_id is None:
                lemma_cls_id = cur_class

            if t == 'p':
                items = n_key_data['p']
            else:
                items = n_key_data['o']

            items.append((item['text'], cur_class))
            if t == 'p':
                end = get_nar_end(item['text'])
                n_key_data['nar_end'][cur_class] = end

regex = []
for val in res_dict:
    cur_group = []
    for tpl in res_dict[val]['p']:
        cur_group.append(tpl[0])

    for tpl in res_dict[val]['o']:
        cur_group.append(tpl[0])

    cur_group = '|'.join(cur_group)
    cur_group = f'(?<_{val}>{cur_group})'
    regex.append(cur_group)

regex = '|'.join(regex)
regex = f"^({regex})+$"


with open(NMB_CLASSES_PATH, 'wb+') as f:
    pickle.dump(numb_cls_dict, f)

with open(NMB_DATA_PATH, 'wb+') as f:
    pickle.dump({
        'regex': regex,
        'lemma_cls_id': lemma_cls_id,
        'numbers': res_dict
    }, f)
