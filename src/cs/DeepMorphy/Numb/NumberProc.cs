using System.Collections.Generic;
using System.Linq;

namespace DeepMorphy.Numb
{
    internal class NumberProc : IMorphProcessor
    {
        public string Key => "numb";
        public bool IgnoreNetworkResult => false;
        
        public IEnumerable<(int tagId, string lemma)> Parse(string word)
        {
            if (!_tryParse(word, 
                out string prefix,
                out string mainWord,
                out List<(int tagId, string text)> curLexeme))
            {
                return null;
            }

            var lemma = curLexeme.First(x => NumbInfo.LemmaTagId.Contains(x.tagId)).text;
            return curLexeme.Where(x => x.text == mainWord).Select(kp => (kp.tagId, lemma));
        }

        public string Lemmatize(string word, int tagId)
        {
            if (!_tryParse(word, 
                out string prefix,
                out string mainWord,
                out List<(int tagId, string text)> curLexeme))
            {
                return null;
            }
            

            var lemma = curLexeme.First(x => NumbInfo.LemmaTagId.Contains(x.tagId)).text;
            return lemma;
        }

        public string Inflect(string word, int wordTag, int resultTag)
        {
            if (!_tryParse(word, 
                out string prefix,
                out string mainWord,
                out List<(int tagId, string text)> curLexeme))
            {
                return null;
            }

            if (curLexeme.All(x => x.tagId != resultTag))
            {
                return null;
            }

            var mainValue = curLexeme.First(x => x.tagId == resultTag).text;
            return $"{prefix}{mainValue}";
        }

        public IEnumerable<(int tagId, string text)> Lexeme(string word, int tagId)
        {
            if (!_tryParse(word,
                out string prefix,
                out string mainWord,
                out List<(int tagId, string text)> curLexeme))
            {
                return null;
            }

            return curLexeme.Select(tpl => (tpl.tagId, $"{prefix}{tpl.text}"));
        }

        private bool _tryParse(string word,
            out string prefix,
            out string mainWord,
            out List<(int tagId, string text)> curLexeme)
        {
            prefix = null;
            mainWord = null;
            string mWord = null;
            curLexeme = null;
            NumbInfo.NumberData numbData = null;
            var match = NumbInfo.NumberRegex.Match(word);
            if (!match.Success)
            {
                return false;
            }

            foreach (var kp in NumbInfo.RegexGroups)
            {
                var grVal = match.Groups[kp.Key];
                if (grVal.Success && grVal.Index + grVal.Length == word.Length)
                {
                    prefix = word.Substring(0, grVal.Index);
                    mWord = grVal.Value;
                    numbData = NumbInfo.NumberDictionary[kp.Value];
                    break;
                }
            }

            if (numbData == null)
            {
                return false;
            }

            mainWord = mWord;
            foreach (var lexemeKp in numbData.Lexemes)
            {
                if (lexemeKp.Value.Any(x => x.text == mWord))
                {
                    curLexeme = lexemeKp.Value;
                    return true;
                }
            }
            
            return false;
        }
    }
}