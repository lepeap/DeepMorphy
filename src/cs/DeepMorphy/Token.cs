using System;
using System.Collections.Generic;
using System.Linq;

namespace DeepMorphy
{
    public sealed class Token
    {
        private Dictionary<string, TagCollection> _grams;
        
        
        internal Token(
            string text,
            TagsCombination[] tagsCombination,
            Dictionary<string, TagCollection> grams,
            string lemma=null
        )
        {
            Text = text;
            Lemma = lemma;
            TagsCombination = tagsCombination;
            _grams = grams;
        }
        public string Text { get; }
        public string Lemma { get; }
        public TagsCombination[] TagsCombination { get; }       
        public TagsCombination BestTagsCombination => TagsCombination?.First();
        public Tag this[string gramKey, string tag]
        {
            get
            {
                if (!_grams.ContainsKey(gramKey))
                    return new Tag(tag, 0);

                var tagVal = _grams[gramKey].Tags
                                            .FirstOrDefault(x=>x.Key==tag);
                if (tagVal==null)
                    return new Tag(tag, 0);
                
                return tagVal;
            }
        }       
        public TagCollection this[string gramKey]
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
            return $"{Text} : {BestTagsCombination}";
        }

        internal Token MakeCopy(string text)
        {
            return new Token(text, TagsCombination, _grams);
        }
    }
}