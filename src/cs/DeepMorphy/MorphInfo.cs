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
        private readonly bool _useEnGrams;
        private Dictionary<string, GramCategory> _gramCats;
        
        internal MorphInfo(
            string text,
            Tag[] tags,
            Dictionary<string, GramCategory> gramCats,
            bool useEnGrams=false
        )
        {
            Text = text;
            Tags = tags;
            _gramCats = gramCats;
            _useEnGrams = useEnGrams;
        }
        
        internal MorphInfo(
            string text,
            Tag[] tags,
            bool useEnGrams
        )
        {
            Text = text;
            Tags = tags;
            _useEnGrams = useEnGrams;
        }
        
        internal MorphInfo MakeCopy()
        {
            if (_gramCats!=null)
                return new MorphInfo(Text, Tags, _gramCats, _useEnGrams);
            
            return new MorphInfo(Text, Tags, _useEnGrams);
        }
        
        internal MorphInfo MakeCopy(string text, Func<string, string> lemmaGen)
        {
            var tagCombs = Tags.Select(t => 
                new Tag(t.GramsDic, 
                    t.Power, 
                    lemmaGen?.Invoke(t.Lemma), 
                    t.ClassIndex)
            ).ToArray();
            return new MorphInfo(text, tagCombs, _gramCats, _useEnGrams);
        }

        private Dictionary<string, GramCategory> GramsCat
        {
            get
            {
                if (_gramCats == null)
                {
                    _gramCats = new Dictionary<string, GramCategory>();
                    foreach (var gram in GramInfo.GramsInfo)
                    {
                        var gramName = _useEnGrams ? gram.KeyEn : gram.KeyRu;
                        var grams = Tags.Select(x => x[gramName])
                            .Where(x => x != null)
                            .Distinct()
                            .ToArray();

                        if (grams.Length == 0)
                        {
                            grams = gram.Classes
                                .Select(x => _useEnGrams ? x.KeyEn : x.KeyRu)
                                .ToArray();
                        }
                        
                        var power = (float)1.0 / grams.Length;
                        _gramCats[gramName] = new GramCategory(grams.Select(x => new Gram(x, power)).ToArray());
                    }
                }

                return _gramCats;
            }
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
                if (!GramsCat.ContainsKey(gramCatKey))
                    return null;
                
                return GramsCat[gramCatKey];
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
                if (!GramsCat.ContainsKey(gramCatKey))
                    return new Gram(gramKey, 0);

                var gramVal = GramsCat[gramCatKey].Grams
                                            .FirstOrDefault(x=>x.Key==gramKey);
                if (gramVal==null)
                    return new Gram(gramKey, 0);
                
                return gramVal;
            }
        }
        
        /// <summary>
        /// Checks if word can have same lexeme
        /// </summary>
        /// <param name="mi">Second word</param>
        /// <returns>True if words can have same lexeme</returns>
        public bool CanBeSameLexeme(MorphInfo mi)
        {
            var l1 = mi.Tags
                       .Select(x => x.Lemma)
                       .ToList();
            l1.Add(mi.Text);
            
            var l2 = Tags
                .Select(x => x.Lemma)
                .ToList();
            l2.Add(Text);

            return l1.Intersect(l2).Any();
        }

        public override string ToString()
        {
            return $"{Text} : {BestTag}";
        }


    }
}