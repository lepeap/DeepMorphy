using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text.RegularExpressions;

namespace DeepMorphy.PreProc
{
    //internal class RegProc : IMorphProcessor
    //{
    //    
    //    private static readonly Regex Reg;
    //    private static readonly string[] Groups;
    //    static RegProc()
    //    {
    //        var tpls = new[]
    //        {
    //            ("romn", @"(?=[MDCLXVI])M*(C[MD]|D?C*)(X[CL]|L?X*)(I[XV]|V?I*)"),
    //            ("int", @"[0-9]+"),
    //            ("punct", @"\p{P}+")
    //        };
    //        Groups = tpls.Select(x => x.Item1).ToArray();
    //        var groups =  tpls.Select(x=> $"(?<{x.Item1}>^{x.Item2}$)");
//
    //        var rezReg = string.Join("|", groups);
    //        rezReg = $"{rezReg}";
    //        Reg = new Regex(rezReg, RegexOptions.Compiled | RegexOptions.IgnoreCase);
    //    }
//
    //    private readonly char[] _availableChars;
    //    private readonly int _minAvailablePersent;
    //    private readonly bool _useEnGrams;
    //    private readonly bool _withLemmatization;
    //    private Dictionary<string, MorphInfo> TokensCache { get; set; } = new Dictionary<string, MorphInfo>();
    //    public RegProc(char[] availableChars, bool useEnGrams, int minAvailablePersent, bool withLemmatization)
    //    {
    //        _availableChars = availableChars;
    //        _minAvailablePersent = minAvailablePersent;
    //        _useEnGrams = useEnGrams;
    //        _withLemmatization = withLemmatization;
    //    }
    //    
    //    public MorphInfo Parse(string word)
    //    {
    //        var match = Reg.Match(word);
    //        if (!match.Success)
    //        {
    //            var availableCount = word.Count(x => _availableChars.Contains(x));
    //            var availablePers = 100 * availableCount / word.Length;
    //            if (availablePers < _minAvailablePersent)
    //                return GetPostToken(word, "unkn");
    //            return null;
    //        }
//
    //        foreach (var group in Groups)
    //        {
    //            var gr = match.Groups[group];
    //            if (gr.Success)
    //                return GetPostToken(word, group);
    //        }
//
    //        return null;
    //    }
    //    
    //    private MorphInfo GetPostToken(string text, string tag)
    //    {
    //        if (TokensCache.ContainsKey(tag))
    //        {
    //            var token = TokensCache[tag];
    //            return token.MakeCopy(text,
    //                                  _withLemmatization ? x => text : (Func<string, string>)null);
    //        }
    //        else
    //        {
    //            var gram = "post";
    //            var tagKey = tag;
    //            if (!_useEnGrams)
    //            {
    //                tag = GramInfo.EnRuDic[tag];
    //                gram = GramInfo.EnRuDic[gram];
    //            }
//
    //            var lemma = _withLemmatization ? text : null;
//
    //
    //            var gramDic = new Dictionary<string, string>() {{gram, tag}};
    //            var token = new MorphInfo(
    //                text,
    //                new []
    //                {
    //                    new Tag(
    //                        new ReadOnlyDictionary<string, string>(gramDic), 
    //                        (float)1.0, lemma
    //                    )
    //                },
    //                new Dictionary<string, GramCategory>()
    //                {
    //                    {gram, new GramCategory(new[]{new Gram(tag, (float)1.0)})}
    //                },
    //                _useEnGrams
    //            );
//
    //            TokensCache[tagKey] = token;
    //            return token;
    //        }
    //        
    //    }
//
    //}
}