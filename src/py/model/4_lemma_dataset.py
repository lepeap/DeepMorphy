import pickle
from tqdm import tqdm
from collections import defaultdict
from utils import CONFIG, save_dataset


VECT_PATH = CONFIG['vect_words_path']
CLS_CLASSES_PATH = CONFIG['cls_classes_path']


def generate(vec_words, main_cls_dic):
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

            if word_y not in vec_words:
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


with open(VECT_PATH, 'rb') as f:
    vwords = pickle.load(f)

with open(CLS_CLASSES_PATH, 'rb') as f:
    cls_dic = pickle.load(f)

generate(vwords, cls_dic)
