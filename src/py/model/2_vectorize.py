import pickle
from tqdm import tqdm
from utils import CONFIG, vectorize_text


WORDS_PATH = CONFIG['dataset_words_path']
VECT_PATH = CONFIG['vect_words_path']
CHARS = CONFIG['chars']
CHARS_INDEXES = {c: index for index, c in enumerate(CHARS)}


def vectorize_words(words_dic):
    vect_dic = {}
    for word in tqdm(words_dic, desc="Vectorizing words"):
        vect_dic[word] = {
            'vect': vectorize_text(word, CHARS, CHARS_INDEXES),
            'forms': words_dic[word]
        }

    return vect_dic


with open(WORDS_PATH, 'rb') as f:
    words_dic = pickle.load(f)

vec_words = vectorize_words(words_dic)

with open(VECT_PATH, 'wb+') as f:
    pickle.dump(vec_words, f)
