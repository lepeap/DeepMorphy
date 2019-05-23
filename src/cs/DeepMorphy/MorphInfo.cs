using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;

namespace DeepMorphy
{
    /// <summary>
    /// Result of morphology analysis for word
    /// </summary>
    public sealed class MorphInfo
    {
        private Dictionary<string, GramCategory> _grams;
        
        internal MorphInfo(
            string text,
            Tag[] tags,
            Dictionary<string, GramCategory> grams
        )
        {
            Text = text;
            Tags = tags;
            _grams = grams;
        }
        
        internal MorphInfo MakeCopy(string text, Func<string, string> lemmaGen)
        {
            var tagCombs = Tags.Select(t => 
                new Tag(t.Grams, 
                    t.Power, 
                    lemmaGen?.Invoke(t.Lemma), 
                    t.ClassIndex)
            ).ToArray();
            return new MorphInfo(text, tagCombs, _grams);
        }
        
        /// <summary>
        /// Text of analyzed word
        /// </summary>
        public string Text { get; }
        
        /// <summary>
        /// Most probable combinations of grammemes
        /// </summary>
        public Tag[] Tags { get; }
        
        /// <summary>
        /// Best combination of grammemes
        /// </summary>
        public Tag BestTag => Tags?.First();
        
        /// <summary>
        /// Checks if any of grammemes combinations has lemma
        /// </summary>
        /// <param name="lemma">Lemma value to check</param>
        /// <returns>True if has</returns>
        public bool HasLemma(string lemma)
        {
            return Tags.Any(x => x.Lemma == lemma);
        }
        
        /// <summary>
        /// Returns probability distribution for grammemes in selected grammatical category
        /// </summary>
        /// <param name="gramKey">Grammatical category key</param>
        public GramCategory this[string gramKey]
        {
            get
            {
                if (!_grams.ContainsKey(gramKey))
                    return null;
                
                return _grams[gramKey];
            }
        }
        
        /// <summary>
        /// Returns probability of grammeme in grammatical category
        /// </summary>
        /// <param name="gramKey">Grammatical category key</param>
        /// <param name="tag">Grammeme key</param>
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

        public override string ToString()
        {
            return $"{Text} : {BestTag}";
        }


    }
}