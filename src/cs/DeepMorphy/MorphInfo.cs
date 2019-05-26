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
        private Dictionary<string, GramCategory> _gramCats;
        
        internal MorphInfo(
            string text,
            Tag[] tags,
            Dictionary<string, GramCategory> gramCats
        )
        {
            Text = text;
            Tags = tags;
            _gramCats = gramCats;
        }
        
        internal MorphInfo MakeCopy(string text, Func<string, string> lemmaGen)
        {
            var tagCombs = Tags.Select(t => 
                new Tag(t.GramsDic, 
                    t.Power, 
                    lemmaGen?.Invoke(t.Lemma), 
                    t.ClassIndex)
            ).ToArray();
            return new MorphInfo(text, tagCombs, _gramCats);
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
        /// Checks if any of tags has grammeme combination
        /// </summary>
        /// <param name="grams">grammemes</param>
        /// <returns>True if has</returns>
        public bool HasCombination(params string[] grams)
        {
            return Tags.Any(x => x.Has(grams));
        }
        
        
        /// <summary>
        /// Checks if any of tags has lemma
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
        /// <param name="gramCatKey">Grammatical category key</param>
        public GramCategory this[string gramCatKey]
        {
            get
            {
                if (!_gramCats.ContainsKey(gramCatKey))
                    return null;
                
                return _gramCats[gramCatKey];
            }
        }
        
        /// <summary>
        /// Returns probability of grammeme in grammatical category
        /// </summary>
        /// <param name="gramCatKey">Grammatical category key</param>
        /// <param name="gramKey">Grammeme key</param>
        public Gram this[string gramCatKey, string gramKey]
        {
            get
            {
                if (!_gramCats.ContainsKey(gramCatKey))
                    return new Gram(gramKey, 0);

                var gramVal = _gramCats[gramCatKey].Grams
                                            .FirstOrDefault(x=>x.Key==gramKey);
                if (gramVal==null)
                    return new Gram(gramKey, 0);
                
                return gramVal;
            }
        }       

        public override string ToString()
        {
            return $"{Text} : {BestTag}";
        }


    }
}