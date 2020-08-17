using System.Collections.Generic;
using System.Linq;

namespace DeepMorphy.WordDict
{
    internal class DictProc : IMorphProcessor
    {
        private readonly Dict _dict;
        public DictProc()
        {
            _dict = new Dict("dict");
        }

        public IEnumerable<(int tagId, string lemma)> Parse(string word)
        {
            var dictEn = _dict.Get(word);
            if (dictEn == null)
            {
                return null;
            }

            return dictEn.SelectMany(x => x).Where(x => TagHelper.LemmasIds.Contains(x.tag))
                .Select(x => (x.tag, x.word));
        }

        public string Inflect(string word, int wordTag, int resultTag)
        {
            throw new System.NotImplementedException();
        }

        public IEnumerable<(int tag, string text)> Lexeme(string word, int tag)
        {
            throw new System.NotImplementedException();
        }
    }
}