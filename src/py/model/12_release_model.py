import logging
import os
import pickle
from shutil import copyfile

from xml.etree.ElementTree import ElementTree
from lxml import etree

from model import RNN
from utils import CONFIG
from tester import Tester


class Releaser:
    def __init__(self):
        self.config = CONFIG
        self.dataset_path = self.config['dict_path']
        self.model_key = self.config['model_key']
        self.chars = self.config['chars']
        self.gram_types = self.config['grammemes_types']
        self.rnn = RNN(True)
        self.tester = Tester()
        self.pd_publish_paths = [
            os.path.join(path, f"frozen_model_{self.model_key}.pb")
            for path in self.config['publish_net_paths']
        ]
        self.xml_publish_paths = [
            os.path.join(path, f"release_{self.model_key}.xml")
            for path in self.config['publish_net_paths']
        ]
        self.xml_gram_paths = [
            os.path.join(path, "grams.xml")
            for path in self.config['publish_gramm_paths']
        ]
        self.xml_numbers_paths = [
            os.path.join(path, "numbers.xml")
            for path in self.config['publish_numbers_paths']
        ]
        self.xml_tags_paths = [
            os.path.join(path, "tags.xml")
            for path in self.config['publish_tags_paths']
        ]
        self.test_result_paths = [
            os.path.join(path, "test_info.txt")
            for path in self.config['test_results_paths']
        ]
        self.publish_dataset_info_paths = [
            os.path.join(path, "dataset_info.txt")
            for path in self.config['publish_dataset_info_paths']
        ]
        self.public_inflect_templates_paths = [
            os.path.join(path, "inflect_templates.xml")
            for path in self.config['public_inflect_templates_paths']
        ]
        self.classes_dic = self.config['main_classes']
        self.rev_classes_dic = {
            self.classes_dic[key]: ",".join([key for key in list(key) if key is not None])
            for key in self.classes_dic
        }
        with open(CONFIG['tags_path'], 'rb') as f:
            self.tags = pickle.load(f)

        with open(CONFIG['numb_data_path'], 'rb') as f:
            self.numb_data = pickle.load(f)

        with open(self.config['inflect_templates_path'], 'rb') as f:
            self.inflect_templates = pickle.load(f)

    def release_model(self):
        pd_release_path, gram_ops, out_ops = self.rnn.release()
        for path in self.pd_publish_paths:
            copyfile(pd_release_path, path)

        self.__release_test_metrics__()
        self.__release_numbers_xml__()
        self.__release_gramm_docs__()
        self.__release_inflect_docs__()
        self.__release_grams_xml__()
        self.__release_tags_xml__()
        self.__release_dataset_info__()
        self.__release_model_xml__(out_ops, gram_ops)

    def __release_test_metrics__(self):
        results = self.tester.test()
        for path in self.test_result_paths:
            with open(path, 'w+') as f:
                f.write(results)

    def __release_model_xml__(self, out_ops, gram_ops):
        root = etree.Element('Root')
        for key in out_ops:
            root.set(key, out_ops[key])

        chars_el = etree.Element('Chars')
        chars_el.set("start_char", str(self.config['start_token']))
        chars_el.set("end_char", str(self.config['end_token']))
        for index, value in enumerate(self.chars):
            char_el = etree.Element("Char")
            char_el.set('index', str(index))
            char_el.set('value', value)
            chars_el.append(char_el)
        root.append(chars_el)

        grams_el = etree.Element('Grams')
        for gram in self.gram_types:
            gram_el = etree.Element("G")
            gram_el.set('key', gram)
            gram_el.set('op', gram_ops[gram]['prob'])
            grams_el.append(gram_el)
        root.append(grams_el)

        inflect_el = etree.Element("Inflect")
        for main_key in self.inflect_templates:
            temp_el = etree.Element("Im")
            temp_el.set('i', str(self.classes_dic[main_key]))
            inflect_el.append(temp_el)
            for form in self.inflect_templates[main_key]:
                form_el = etree.Element("I")
                form_el.set('i', str(self.classes_dic[form]))
                temp_el.append(form_el)
        root.append(inflect_el)

        tree = ElementTree(root)
        for path in self.xml_publish_paths:
            with open(path, 'wb+') as f:
                tree.write(f, xml_declaration=True, encoding='utf-8')

        logging.info("Model released")

    def __release_grams_xml__(self):
        nn_types = self.config['grammemes_types']
        dict_post_types = self.config['dict_post_types']
        other_types = self.config['other_post_types']
        root = etree.Element('Grams')
        for gram in nn_types:
            gram_el = etree.Element("G")
            gram_el.set('index', str(nn_types[gram]['index']))
            gram_el.set('key_en', gram)
            gram_el.set('key_ru', nn_types[gram]['key_ru'])
            root.append(gram_el)
            gr_dic = nn_types[gram]['classes']
            for key_en in gr_dic:
                item = gr_dic[key_en]
                cls_el = etree.Element("C")
                cls_el.set('key_en', key_en)
                cls_el.set('key_ru', str(item['key_ru']))
                cls_el.set('nn_index', str(item['index']))
                gram_el.append(cls_el)

            if gram == "post":
                for key_en in dict_post_types:
                    item = dict_post_types[key_en]
                    cls_el = etree.Element("C")
                    cls_el.set('key_en', key_en)
                    cls_el.set('key_ru', str(item['key_ru']))
                    gram_el.append(cls_el)

                for key_en in other_types:
                    item = other_types[key_en]
                    cls_el = etree.Element("C")
                    cls_el.set('key_en', key_en)
                    cls_el.set('key_ru', str(item['key_ru']))
                    gram_el.append(cls_el)

        tree = ElementTree(root)
        for path in self.xml_gram_paths:
            with open(path, 'wb+') as f:
                tree.write(f, xml_declaration=True, encoding='utf-8')

    def __release_numbers_xml__(self):
        root = etree.Element('NumbData')
        root.set("reg", self.numb_data['regex'])
        root.set("l", ','.join([str(i) for i in self.numb_data['lemma_cls_ids']]))
        for val in self.numb_data['numbers']:
            n_el = etree.Element("N")
            n_el.set('v', str(val))
            for tp in self.numb_data['numbers'][val]:
                if tp == 'nar_end' or tp == 'lemma':
                    continue

                for tpl in self.numb_data['numbers'][val][tp]:
                    w_el = etree.Element("W")
                    w_el.set('t', tpl[0])
                    w_el.set('i', str(tpl[1]))
                    w_el.set('k', tp)
                    n_el.append(w_el)

            nar_ends = self.numb_data['numbers'][val]['nar_end']
            for cls in nar_ends:
                w_el = etree.Element("E")
                w_el.set('t', nar_ends[cls])
                w_el.set('i', str(cls))
                n_el.append(w_el)

            root.append(n_el)

        tree = ElementTree(root)
        for path in self.xml_numbers_paths:
            with open(path, 'wb+') as f:
                tree.write(f, xml_declaration=True, encoding='utf-8')

    def __release_tags_xml__(self):
        root = etree.Element('Tags')
        for tag in self.tags:
            val = self.tags[tag]
            cls_el = etree.Element("T")
            cls_el.set('i', str(val['i']))
            cls_el.set('v', ",".join([key if key is not None else '' for key in tag]))
            cls_el.set('p', val['p'])
            cls_el.set('o', str(val['o']))

            if val['l']:
                cls_el.set('l', '1')
            root.append(cls_el)

        tree = ElementTree(root)
        for path in self.xml_tags_paths:
            with open(path, 'wb+') as f:
                tree.write(f, xml_declaration=True, encoding='utf-8')

    def __release_dataset_info__(self):
        doc = etree.iterparse(self.dataset_path, events=('start', 'end'))
        itr = iter(doc)
        event, element = next(itr)
        while not (event == 'start' and element.tag == 'dictionary'):
            pass

        version = element.attrib['version']
        revision = element.attrib['revision']
        for path in self.publish_dataset_info_paths:
            with open(path, 'w+') as f:
                f.write(f"dictionary\nversion={version}\nrevision={revision}")

    def __release_gramm_docs__(self):
        mds = [
            "# Поддерживамые грамматические категории и граммемы",
            "В DeepMorphy используется слегка измененное подмножество граммем и грамматичеких категорий из словарей [OpenCorpora](http://opencorpora.org/dict.php?act=gram)."
        ]

        for gram_cat_key in self.gram_types:
            gram_cat = self.gram_types[gram_cat_key]
            mds.append(f"- **{gram_cat['name'].capitalize()}** (ru='{gram_cat['key_ru']}', en='{gram_cat_key}') :")

            classes = dict(gram_cat['classes'])
            if gram_cat_key == 'post':
                classes.update(self.config['dict_post_types'])
                classes.update(self.config['other_post_types'])

            for gram in classes:
                gram_obj = classes[gram]
                mds.append(f"    - {gram_obj['name_ru']} (ru='{gram_obj['key_ru']}',en='{gram}')")

        mds = "\n".join(mds)
        with open(self.config['publish_gram_doc_path'], 'w+') as f:
            f.write(mds)

    def __release_inflect_docs__(self):
        mds = [
            "# Список поддерживаемых словоизменений",
            "Словоизменение возможно только в рамках выделенных жирным категорий:"
        ]

        post_index = self.gram_types['post']['index']
        gndr_index = self.gram_types['gndr']['index']

        en_ru_dict = {}
        for gram_cat in self.gram_types:
            for cls in self.gram_types[gram_cat]['classes']:
                cls_data = self.gram_types[gram_cat]['classes'][cls]
                en_ru_dict[cls] = cls_data['key_ru']

        def create_tag_text(tag):
            tag_text = [en_ru_dict[key] for key in list(tag) if key is not None]
            tag_text = ",".join(tag_text)
            return f"    - {tag_text}"

        for main_tpl in sorted(self.inflect_templates):
            post = main_tpl[post_index]
            gndr = main_tpl[gndr_index]
            if post == "infn":
                header_text = "Глаголы и глагольные формы"
            elif post == "adjf":
                header_text = "Прилагательные"
            elif post == "noun" and gndr == 'masc':
                header_text = "Существительные мужского рода"
            elif post == "noun" and gndr == 'femn':
                header_text = "Существительные женского рода"
            elif post == "noun" and gndr == 'neut':
                header_text = "Существительные среднего рода"
            elif post == "noun" and gndr == 'msf':
                header_text = "Существительные общего рода"
            else:
                raise NotImplemented()

            mds.append(f"- **{header_text}**:")
            items = [(item, self.tags[item]['o']) for item in self.inflect_templates[main_tpl]]
            items.append((main_tpl, self.tags[main_tpl]['o']))
            tags = sorted(items, key=lambda x: x[1], reverse=True)
            for tag in tags:
                mds.append(create_tag_text(tag[0]))

        mds = "\n".join(mds)
        with open(self.config['publish_inflect_doc_path'], 'w+') as f:
            f.write(mds)

    @staticmethod
    def __build_bad_words__(tester):
        words = tester.get_bad_words()
        logging.info(f"Wrong words count {len(words)}")
        with open(os.path.join("wrong_words.pkl"), 'wb+') as f:
            pickle.dump(words, f)


if __name__ == "__main__":
    tester = Releaser()
    tester.release_model()
