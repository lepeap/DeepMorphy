import pickle
import numpy as np
from tqdm import tqdm
from utils import CONFIG

CHARS = CONFIG['chars']
END_TOKEN = CONFIG['end_token']
MAX_WORD_SIZE = CONFIG['max_word_size']
WORDS_PATH = CONFIG['dataset_words_path']
VECT_PATH = CONFIG['vect_words_path']
CHARS_INDEXES = {c: index for index, c in enumerate(CHARS)}


def vectorize_text(text):
    word_vect = np.full((MAX_WORD_SIZE,), END_TOKEN, dtype=np.int32)
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


with open(WORDS_PATH, 'rb') as f:
    words_dic = pickle.load(f)

vec_words = vectorize_words(words_dic)

with open(VECT_PATH, 'wb+') as f:
    pickle.dump(vec_words, f)


