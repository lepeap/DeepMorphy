using System.Collections.Generic;
using System.Linq;

namespace DeepMorphy.WordDict
{
    internal class DictProc : IMorphProcessor
    {
        private readonly Dict _dict;

        public DictProc(string dictKey)
        {
            _dict = new Dict(dictKey);
        }

        public string Key => "dict";
        
        public bool IgnoreNetworkResult => false;

        public IEnumerable<(int tagId, string lemma)> Parse(string word)
        {
            return _dict.Parse(word).Select(x => (x.TagId, x.Lemma));
        }

        public string Lemmatize(string word, int tagId)
        {
            var dictEn = _dict.Parse(word);
            if (dictEn == null)
            {
                return null;
            }

            var res = dictEn.Where(x => x.TagId == tagId).ToArray();
            if (res.Length == 0)
            {
                return null;
            }

            return res[0].Lemma;
        }

        public string Inflect(string word, int wordTag, int resultTag)
        {
            var lexeme = _dict.Lexeme(word, wordTag);
            if (lexeme == null)
            {
                return null;
            }

            return lexeme.FirstOrDefault(x => x.TagId == resultTag)?.Lemma;
        }

        public IEnumerable<(int tagId, string text)> Lexeme(string word, int tag)
        {
            var lexeme = _dict.Lexeme(word, tag);
            if (lexeme == null)
            {
                return null;
            }

            return lexeme.Select(x => (x.TagId, x.Text));
        }
    }
}