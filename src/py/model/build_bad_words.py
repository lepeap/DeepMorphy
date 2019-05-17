import os
import pickle
from tester import Tester

tester = Tester()

words = []
def load_words(type):
    path = os.path.join(tester.config['dataset_path'], f"lemma_{type}_dataset.pkl")
    with open(path, 'rb') as f:
        words.extend(pickle.load(f))

load_words("test")
#load_words("valid")
#load_words("train")

words = [
    word
    for word in words
    if all([c in tester.config['chars'] for c in word['x_src']])
]

words_to_parse = [
    (word['x_src'], word['main_cls'])
    for word in words
]

lemmas = tester.get_test_lemmas(words_to_parse)

wrong_words = []
for index, lem in enumerate(lemmas):
    et_word = words[index]
    if lem != et_word['y_src']:
        wrong_words.append(et_word['x_src'])

print(f"Wrong lemmas count: {len(wrong_words)}")
with open(os.path.join("wrong_words.pkl"), 'wb+') as f:
    pickle.dump(wrong_words, f)
