import pickle
from tqdm import tqdm

from utils import CONFIG, save_dataset


MIN_WORD_SIZE = CONFIG['min_word_size']
PREFIX_FILTER_LENGTH = CONFIG['prefix_filter_length']
VECT_PATH = CONFIG['vect_words_path']
CLS_CLASSES_PATH = CONFIG['cls_classes_path']
GRAMMEMES_TYPES = CONFIG['grammemes_types']
TEMPLATES_PATH = CONFIG['inflect_templates_path']
IGNORE_TAGS = CONFIG['inflect_ignore_tags']


def create_forms_dict(vect_words):
    root_posts = ['noun', 'infn', 'adjf']
    forms_dict = {}
    for word in vect_words:
        for form in vect_words[word]['forms']:
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
    for key in tqdm(forms_dict, desc="Generating dataset"):
        item = forms_dict[key]
        root = item['root']
        x_cls = cls_dic[root['main']]
        x, x_len = vect_words[root['text']]['vect']
        prefix_filter = root['text'][:PREFIX_FILTER_LENGTH]
        prefix_filter_e = prefix_filter.replace('ё', 'е')
        if MIN_WORD_SIZE > len(root['text']):
            continue

        form_dict = {}
        for form in item['items']:
            if MIN_WORD_SIZE > len(form['text']):
                continue

            if 'ad_tags' in form and any([tag for tag in IGNORE_TAGS if tag in form['ad_tags']]):
                #tqdm.write("Ignore form {0} for {1} by tags {2}".format(form['text'], root['text'], form['ad_tags']))
                continue

            if not (form['text'].startswith(prefix_filter) or
                    form['text'].replace('ё', 'е').startswith(prefix_filter_e)):
                #tqdm.write("Ignore form {0} for {1}".format(form['text'], root['text']))
                continue

            y_cls = cls_dic[form['main']]
            if y_cls in form_dict and form_dict[y_cls]['index'] < form['index']:
                #tqdm.write("Ignore duplicate form {0} [{1}] for {2} ".format(form['text'], form_dict[y_cls]['text'], root['text']))
                continue

            form_dict[y_cls] = form

        for y_cls in form_dict:
            form = form_dict[y_cls]
            y, y_len = vect_words[form['text']]['vect']
            if y_cls not in rez_dict:
                rez_dict[y_cls] = []

            rez_dict[y_cls].append(dict(
                id=form['inflect_id'],
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
with open(TEMPLATES_PATH, 'wb+') as f:
    pickle.dump(templates, f)

generate_dataset(forms_dict, vwords, cls_dic)
