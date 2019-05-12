import os
import logging
from shutil import copyfile
from model import RNN
from utils import config
from lxml import etree
from xml.etree.ElementTree import ElementTree

CONFIG = config()
MODEL_KEY = CONFIG['model_key']
RELEASE_PATH = CONFIG['publish_net_path']
CHARS = CONFIG['chars']
GRAM_TYPES = CONFIG['grammemes_types']

rnn = RNN(True)
pd_publish_path = os.path.join(RELEASE_PATH, f"frozen_model_{MODEL_KEY}.pb")
pd_release_path, classes_dic, gram_ops, out_ops = rnn.release()
copyfile(pd_release_path, pd_publish_path)


xml_path = os.path.join(RELEASE_PATH, f"release_{MODEL_KEY}.xml")
root = etree.Element('Root')
for key in out_ops:
    root.set(key, out_ops[key])

chars_el = etree.Element('Chars')
chars_el.set("start_char", str(CONFIG['start_token']))
chars_el.set("end_char", str(CONFIG['end_token']))


for index, value in enumerate(CHARS):
    char_el = etree.Element("Char")
    char_el.set('index', str(index))
    char_el.set('value', value)
    chars_el.append(char_el)

root.append(chars_el)

grams_el = etree.Element('Grams')
for gram in GRAM_TYPES:
    gram_el = etree.Element("G")
    gram_el.set('key', gram)
    gram_el.set('op', gram_ops[gram]['prob'])
    grams_el.append(gram_el)

root.append(grams_el)

classes_el = etree.Element('Classes')
for cls in classes_dic:
    cls_el = etree.Element("C")
    cls_el.set('i', str(classes_dic[cls]))
    cls_el.set('v', ",".join([key if key is not None else '' for key in cls]))

    classes_el.append(cls_el)

root.append(classes_el)

tree = ElementTree(root)
tree.write(open(xml_path, 'wb+'), xml_declaration=True, encoding='utf-8')

logging.info("Model released")