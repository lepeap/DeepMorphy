using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace DeepMorphy.Numb
{
    internal class NumberProc : IMorphProcessor
    {
        private readonly bool _useEnTags;
        private readonly bool _withLemmatization;
        private readonly Dictionary<int, ReadOnlyDictionary<string, string>> _tagsDic;

        public NumberProc(bool useEnTags, bool withLemmatization)
        {
            _useEnTags = useEnTags;
            _tagsDic = useEnTags ? TagHelper.TagsEnDic : TagHelper.TagsRuDic;
            _withLemmatization = withLemmatization;
        }

        public IEnumerable<MorphInfo> Parse(IEnumerable<string> words)
        {
            foreach (var word in words)
            {
                if (!_tryParse(word, out string prefix, 
                                     out string mainWord,
                                     out Dictionary<int, string> curLexemeDic))
                {
                    yield return null;
                    continue;
                }

                var items = curLexemeDic.Where(x => x.Value == mainWord).ToArray();
                var lemma = curLexemeDic[NumbInfo.LemmaTagId];
                var power = (float)1.0 / items.Length;
                lemma = _withLemmatization ? $"{prefix}{lemma}" : null;
                var tags = items.Select(kp => new Tag(_tagsDic[kp.Key], power, lemma, kp.Key)).ToArray();
                yield return new MorphInfo(word, tags, _useEnTags);
            }
        }

        public IEnumerable<string> Inflect(IEnumerable<(string word, Tag wordTag, Tag resultTag)> tasks)
        {
            foreach (var task in tasks)
            {
                if (!_tryParse(task.word, out string prefix, 
                    out string mainWord,
                    out Dictionary<int, string> curLexemeDic))
                {
                    yield return null;
                    continue;
                }
                
                var tagIndex = task.resultTag.TagIndex;
                if (!tagIndex.HasValue || !curLexemeDic.ContainsKey(tagIndex.Value))
                {
                    yield return null;
                    continue;
                }

                var mainValue = curLexemeDic.FirstOrDefault(x => x.Key == task.resultTag.TagIndex).Value;
                yield return $"{prefix}{mainValue}";
            }
        }

        public IEnumerable<(Tag tag, string text)> Lexeme(string word, Tag tag)
        {
            if (!_tryParse(word, 
                           out string prefix, 
                           out string mainWord,
                           out Dictionary<int, string> curLexemeDic))
            {
                return null;
            }
            var lemma = $"{prefix}{curLexemeDic[NumbInfo.LemmaTagId]}";
            return curLexemeDic.Select(kp => (new Tag(_tagsDic[kp.Key], (float) 1.0, lemma, kp.Key), $"{prefix}{kp.Value}"));
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