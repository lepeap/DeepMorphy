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

        public IEnumerable<(int tagId, string lemma)> Parse(string word)
        {
            var match = Reg.Match(word);
            if (match.Success)
            {
                var gr = Groups.Single(x => match.Groups[x].Success);
                yield return (GroupToClassDic[gr], word);
                yield break;
            }

            var availableCount = word.Count(x => _availableChars.Contains(x));
            var availablePers = 100 * availableCount / word.Length;
            if (availablePers < _minAvailablePersent)
            {
                yield return (GroupToClassDic["unkn"], word);
            }
        }

        public string Inflect(string word, int wordTag, int resultTag)
        {
            return word;
        }

        public IEnumerable<(int tag, string text)> Lexeme(string word, int tag)
        {
            yield return (tag, word);
        }
    }
}