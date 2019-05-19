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
            Tag[] tags,
            Dictionary<string, GramCategory> grams
        )
        {
            Text = text;
            Tags = tags;
            _grams = grams;
        }
        public string Text { get; }
        public Tag[] Tags { get; }       
        public Tag BestTag => Tags?.First();

        public bool HasLemma(string lemma)
        {
            return Tags.Any(x => x.Lemma == lemma);
        }
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
            var tagCombs = Tags.Select(t => 
                                               new Tag(t.Grams, 
                                                                   t.Power, 
                                                                   lemmaGen?.Invoke(t.Lemma), 
                                                                   t.ClassIndex)
                ).ToArray();
            return new Token(text, tagCombs, _grams);
        }
    }
}