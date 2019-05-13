using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;

namespace DeepMorphy
{
    public sealed class Token
    {
        private Dictionary<string, GramCategory> _grams;
        
        
        internal Token(
            string text,
            Tag[] tag,
            Dictionary<string, GramCategory> grams,
            string lemma=null
        )
        {
            Text = text;
            Lemma = lemma;
            Tag = tag;
            _grams = grams;
        }
        public string Text { get; }
        public string Lemma { get; }
        public Tag[] Tag { get; }       
        public Tag BestTag => Tag?.First();
        public Gram this[string gramKey, string tag]
        {
            get
            {
                if (!_grams.ContainsKey(gramKey))
                    return new Gram(tag, 0);

                var tagVal = _grams[gramKey].Grams
                                            .FirstOrDefault(x=>x.Key==tag);
                if (tagVal==null)
                    return new Gram(tag, 0);
                
                return tagVal;
            }
        }       
        public GramCategory this[string gramKey]
        {
            get
            {
                if (!_grams.ContainsKey(gramKey))
                    return null;
                
                return _grams[gramKey];
            }
        }
        public override string ToString()
        {
            return $"{Text} : {BestTag}";
        }

        internal Token MakeCopy(string text, Func<string, string> lemmaGen)
        {
            var tagCombs = Tag.Select(t => 
                                               new Tag(t.Tags, 
                                                                   t.Power, 
                                                                   lemmaGen?.Invoke(t.Lemma), 
                                                                   t.ClassIndex)
                ).ToArray();
            return new Token(text, tagCombs, _grams);
        }
    }
}