from pymorphy2 import MorphAnalyzer


morph = MorphAnalyzer()

rez = morph.parse("большой")

print()