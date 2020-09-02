using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace DeepMorphy
{
    internal class RegProc : IMorphProcessor
    {
        private static readonly Regex Reg;
        private static readonly string[] Groups;
        private static Dictionary<string, int> GroupToClassDic = new Dictionary<string, int>();

        static RegProc()
        {
            var tpls = new[]
            {
                ("romn", @"(?=[MDCLXVI])M*(C[MD]|D?C*)(X[CL]|L?X*)(I[XV]|V?I*)"),
                ("int", @"[0-9]+"),
                ("punct", @"\p{P}+")
            };
            Groups = tpls.Select(x => x.Item1).ToArray();
            var groups = tpls.Select(x => $"(?<{x.Item1}>^{x.Item2}$)");
            var rezReg = string.Join("|", groups);
            rezReg = $"{rezReg}";
            Reg = new Regex(rezReg, RegexOptions.Compiled | RegexOptions.IgnoreCase);
            GroupToClassDic = tpls.ToDictionary(x => x.Item1,
                x => TagHelper.TagsEnDic.First(t => t.Value["post"] == x.Item1).Key);
            GroupToClassDic["unkn"] = TagHelper.TagsEnDic.First(t => t.Value["post"] == "unkn").Key;
        }
        
        private readonly char[] _availableChars;
        private readonly int _minAvailablePersent;
        
        public RegProc(char[] availableChars, int minAvailablePersent)
        {
            _availableChars = availableChars;
            _minAvailablePersent = minAvailablePersent;
        }
        
        public string Key => "reg";
        public bool IgnoreNetworkResult => true;

        public IEnumerable<(int tagId, string lemma)> Parse(string word)
        {
            var match = Reg.Match(word);
            if (match.Success)
            {
                var gr = Groups.Single(x => match.Groups[x].Success);
                return (GroupToClassDic[gr], word).Yield();
            }
            
            if (_isUnknown(word))
            {
                return (GroupToClassDic["unkn"], word).Yield();
            }

            return null;
        }

        public string Lemmatize(string word, int tagId)
        {
            if (Reg.IsMatch(word) || _isUnknown(word))
            {
                return word;
            }

            return null;
        }

        public string Inflect(string word, int wordTag, int resultTag)
        {
            if (Reg.IsMatch(word) || _isUnknown(word))
            {
                return word;
            }

            return null;
        }

        public IEnumerable<(int tagId, string text)> Lexeme(string word, int tag)
        {
            if (Reg.IsMatch(word) || _isUnknown(word))
            {
                return new[]{ (tag, word) };
            }

            return null;
        }

        private bool _isUnknown(string word)
        {
            var availableCount = word.Count(x => _availableChars.Contains(x));
            var availablePers = 100 * availableCount / word.Length;
            return availablePers < _minAvailablePersent;
        }
    }
}