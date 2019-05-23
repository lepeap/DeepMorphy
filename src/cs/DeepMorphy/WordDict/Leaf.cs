using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;

namespace DeepMorphy.WordDict
{
    class Leaf
    {
        internal struct LeafResult
        {
            public LeafResult(string[] tags, string lemma=null)
            {
                Lemma = lemma;
                Tags = tags;
            }
            public string Lemma { get; }
            public string[] Tags { get; }
        }
        
        private readonly bool _useEnGrams;
        public Leaf(){}
        public Leaf(string text, bool useEnGrams)
        {
            Text = text;
            _useEnGrams = useEnGrams;
        }
        
        private Dictionary<char, Leaf> _leaves = new Dictionary<char, Leaf>();
        private List<LeafResult> _results = new List<LeafResult>();
        
        public string Text { get; set; }
        public char Char { get; set; }
        public bool HasResults => _results.Count > 0;


        private MorphInfo _morphInfo;
        public MorphInfo MorphInfo
        {
            get
            {
                if (_morphInfo == null)
                {
                    var combs = _results.Select(x => 
                            new Tag(x.Tags.Where(y => y != null).ToArray(), 
                                                (float)1.0 / _results.Count,
                                                x.Lemma)
                    ).ToArray();

                    var gDic = new Dictionary<string, GramCategory>();
                    foreach (var gram in GramInfo.GramsInfo)
                    {
                        var gramName = _useEnGrams ? gram.KeyEn : gram.KeyRu;
                        var tags = _results.Select(x => x.Tags[gram.Index])
                                           .Where(x => x != null)
                                           .ToArray();
                        var lemmas = _results.Select(x => x.Lemma);
                                             
                        if (tags.Length==0)
                            continue;
                        
                        var power = (float)1.0 / tags.Length;
                        gDic[gramName] = new GramCategory(tags.Select(x => new Gram(x, power)).ToArray());
                    }
                    _morphInfo = new MorphInfo(Text, combs, gDic);
                }
                return _morphInfo;
            }
        }

        public void AddLeaf(Leaf l)
        {
            _leaves[l.Char] = l;
        }

        public void AddResult(LeafResult lr)
        {
            _results.Add(lr);
        }

        public override string ToString()
        {
            return $"{Char} : {_leaves.Count}";
        }

        public Leaf this[char c]
        {
            get
            {
                if (!_leaves.ContainsKey(c))
                    return null;

                return _leaves[c];
            }
        }
    }
}