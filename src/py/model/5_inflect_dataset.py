import pickle
from utils import get_grams_info, CONFIG, save_dataset


VECT_PATH = CONFIG['vect_words_path']
CLS_CLASSES_PATH = CONFIG['cls_classes_path']
GRAMMEMES_TYPES = CONFIG['grammemes_types']
SRC_CONVERT, CLASSES_INDEXES = get_grams_info(CONFIG)


def create_forms_dict(vect_words):
    root_posts = ['noun', 'infn', 'adjf']
    forms_dict = {}
    for word in vect_words:
        for form in vect_words[word]['forms']:
            if isinstance(form['id'], tuple):
                continue

            is_main = 'inflect_id' not in form
            key = form['id'] if is_main else form['inflect_id']
            if key not in forms_dict:
                forms_dict[key] = dict(root=None, items=[])

            form_dict = forms_dict[key]
            if is_main:
                form_dict['root'] = form
            else:
                form_dict['items'].append(form)

    forms_dict = {
        key: forms_dict[key]
        for key in forms_dict if forms_dict[key]['root'] is not None and forms_dict[key]['root']['post'] in root_posts
    }
    return forms_dict


def create_templates(forms_dict):
    templates = dict()
    for key in forms_dict:
        item = forms_dict[key]
        root = item['root']
        if root['main'] not in templates:
            templates[root['main']] = set()

        for form in item['items']:
            templates[root['main']].add(form['main'])

    return templates


def generate_dataset(forms_dict, vect_words, cls_dic):
    rez_dict = {}
    for key in forms_dict:
        item = forms_dict[key]
        root = item['root']
        x_cls = cls_dic[root['main']]
        x, x_len = vect_words[root['text']]['vect']

        for form in item['items']:
            y_cls = cls_dic[form['main']]
            y, y_len = vect_words[form['text']]['vect']
            if y_cls not in rez_dict:
                rez_dict[y_cls] = []

            rez_dict[y_cls].append(dict(
                x_src=root['text'],
                x=x,
                x_cls=x_cls,
                x_len=x_len,
                y_src=form['text'],
                y=y,
                y_cls=y_cls,
                y_len=y_len
            ))

    save_dataset(rez_dict, 'inflect')


with open(VECT_PATH, 'rb') as f:
    vwords = pickle.load(f)

with open(CLS_CLASSES_PATH, 'rb') as f:
    cls_dic = pickle.load(f)

forms_dict = create_forms_dict(vwords)
templates = create_templates(forms_dict)
generate_dataset(forms_dict, vwords, cls_dic)
