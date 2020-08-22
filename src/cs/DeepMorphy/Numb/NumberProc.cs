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
                out Dictionary<int, string> curLexemeDic))
            {
                return null;
            }

            var lemma = curLexemeDic.First(x => NumbInfo.LemmaTagId.Contains(x.Key)).Value;
            return curLexemeDic.Where(x => x.Value == mainWord).Select(kp => (kp.Key, lemma));
        }

        public string Lemmatize(string word, int tagId)
        {
            if (!_tryParse(word, 
                out string prefix,
                out string mainWord,
                out Dictionary<int, string> curLexemeDic))
            {
                return null;
            }

            var lemma = curLexemeDic.First(x => NumbInfo.LemmaTagId.Contains(x.Key)).Value;
            return lemma;
        }

        public string Inflect(string word, int wordTag, int resultTag)
        {
            if (!_tryParse(word, 
                out string prefix,
                out string mainWord,
                out Dictionary<int, string> curLexemeDic))
            {
                return null;
            }

            if (!curLexemeDic.ContainsKey(resultTag))
            {
                return null;
            }

            var mainValue = curLexemeDic.FirstOrDefault(x => x.Key == resultTag).Value;
            return $"{prefix}{mainValue}";
        }

        public IEnumerable<(int tagId, string text)> Lexeme(string word, int tagId)
        {
            if (!_tryParse(word,
                out string prefix,
                out string mainWord,
                out Dictionary<int, string> curLexemeDic))
            {
                return null;
            }

            return curLexemeDic.Select(kp => (kp.Key, $"{prefix}{kp.Value}"));
        }

        private bool _tryParse(string word,
            out string prefix,
            out string mainWord,
            out Dictionary<int, string> curLexemeDic)
        {
            prefix = null;
            mainWord = null;
            string mWord = null;
            curLexemeDic = null;
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
            if (numbData.Ordinal.Where(x => x.Value == mWord).Any())
            {
                curLexemeDic = numbData.Ordinal;
                return true;
            }

            if (numbData.Quantitative.Where(x => x.Value == mWord).Any())
            {
                curLexemeDic = numbData.Quantitative;
                return true;
            }

            return false;
        }
    }
}