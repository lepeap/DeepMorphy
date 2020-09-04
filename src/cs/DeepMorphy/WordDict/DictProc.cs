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
            if (TagHelper.IsLemma(tagId))
            {
                return word;
            }
            
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
            if (wordTag == resultTag)
            {
                return word;
            }
            
            var lexeme = _dict.Lexeme(word, wordTag);
            if (lexeme == null)
            {
                return null;
            }

            var results = lexeme.Where(x => x.TagId == resultTag).ToArray();
            if (results.Length == 0)
            {
                return null;
            }
            
            if (results.Length == 1)
            {
                return results[0].Text;
            }
            
            var result = results.FirstOrDefault(x => x.ReplaceOther)?.Text;
            return result;
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