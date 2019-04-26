import logging
from lxml import etree
from xml.etree.ElementTree import ElementTree
from utils import config

CONFIG = config()
RELEASE_PATH = CONFIG['publish_gramms_path']
NN_TYPES = CONFIG['grammemes_types']
DICT_POST_TYPES = CONFIG['dict_post_types']
OTHER_TYPES = CONFIG['other_post_types']

root = etree.Element('Grams')
for gram in NN_TYPES:
    gram_el = etree.Element("G")
    gram_el.set('index', str(NN_TYPES[gram]['index']))
    gram_el.set('key_en', gram)
    gram_el.set('key_ru', NN_TYPES[gram]['key_ru'])
    root.append(gram_el)
    gr_dic = NN_TYPES[gram]['classes']
    for key_en in gr_dic:
        item = gr_dic[key_en]
        cls_el = etree.Element("C")
        cls_el.set('key_en', key_en)
        cls_el.set('key_ru', str(item['key_ru']))
        cls_el.set('nn_index', str(item['index']))
        gram_el.append(cls_el)

    if gram == "post":
        for key_en in DICT_POST_TYPES:
            item = DICT_POST_TYPES[key_en]
            cls_el = etree.Element("C")
            cls_el.set('key_en', key_en)
            cls_el.set('key_ru', str(item['key_ru']))
            gram_el.append(cls_el)

        for key_en in OTHER_TYPES:
            item = OTHER_TYPES[key_en]
            cls_el = etree.Element("C")
            cls_el.set('key_en', key_en)
            cls_el.set('key_ru', str(item['key_ru']))
            gram_el.append(cls_el)

tree = ElementTree(root)
tree.write(open(RELEASE_PATH, 'wb+'), xml_declaration=True, encoding='utf-8')
logging.info("Gramms released")