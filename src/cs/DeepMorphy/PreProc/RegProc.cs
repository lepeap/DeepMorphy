using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace DeepMorphy.PreProc
{
    class RegProc : IPreProcessor
    {
        
        private static readonly Regex Reg;
        private static readonly string[] Groups;
        static RegProc()
        {
            var tpls = new[]
            {
                ("romn", @"(?=[MDCLXVI])M*(C[MD]|D?C*)(X[CL]|L?X*)(I[XV]|V?I*)"),
                ("int", @"[0-9]+"),
                ("punct", @"\p{P}+")
            };
            Groups = tpls.Select(x => x.Item1).ToArray();
            var groups =  tpls.Select(x=> $"(?<{x.Item1}>^{x.Item2}$)");

            var rezReg = string.Join("|", groups);
            rezReg = $"{rezReg}";
            Reg = new Regex(rezReg, RegexOptions.Compiled | RegexOptions.IgnoreCase);
        }

        private readonly char[] _availableChars;
        private int _minAvailablePersent;
        private bool _useEnTags;
        private bool _withLemmatization;
        private Dictionary<string, Token> _tokensCache { get; set; } = new Dictionary<string, Token>();
        public RegProc(char[] availableChars, bool useEnTags, int minAvailablePersent, bool withLemmatization)
        {
            _availableChars = availableChars;
            _minAvailablePersent = minAvailablePersent;
            _useEnTags = useEnTags;
            _withLemmatization = withLemmatization;
        }
        
        public Token Parse(string word)
        {
            var match = Reg.Match(word);
            if (!match.Success)
            {
                var availableCount = word.Count(x => _availableChars.Contains(x));
                var availablePers = 100 * availableCount / word.Length;
                if (availablePers < _minAvailablePersent)
                    return GetPostToken(word, "unkn");
                return null;
            }

            foreach (var group in Groups)
            {
                var gr = match.Groups[group];
                if (gr.Success)
                    return GetPostToken(word, group);
            }

            return null;
        }
        
        private Token GetPostToken(string text, string tag)
        {
            if (_tokensCache.ContainsKey(tag))
            {
                var token = _tokensCache[tag];
                return token.MakeCopy(text,
                                      _withLemmatization ? x => text : (Func<string, string>)null);
            }
            else
            {
                var gram = "post";
                var tagKey = tag;
                if (!_useEnTags)
                {
                    tag = GramInfo.EnRuDic[tag];
                    gram = GramInfo.EnRuDic[gram];
                }

                var lemma = _withLemmatization ? text : null;
                
                var token = new Token(
                    text,
                    new []{new Tag(new[]{tag}, (float)1.0, lemma)},
                    new Dictionary<string, GramCategory>()
                    {
                        {gram, new GramCategory(new[]{new Gram(tag, (float)1.0)})}
                    }
                );

                _tokensCache[tagKey] = token;
                return token;
            }
            
        }

    }
}