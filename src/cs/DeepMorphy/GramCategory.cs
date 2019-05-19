using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace DeepMorphy
{
    /// <summary>
    /// Grammatical category, defines probability for each grammeme in category for current word
    /// </summary>
    public sealed class GramCategory
    {
        private Gram[] _grams;
        internal GramCategory(Gram[] grams)
        {
            _grams = grams;
        }
        
        /// <summary>
        /// Best grammeme in category for current word
        /// </summary>
        public Gram BestGram => _grams[0];
        
        /// <summary>
        /// All grammemes with probabilities for current word
        /// </summary>
        public IEnumerable<Gram> Grams => _grams;
        
        /// <summary>
        /// Returns grammeme and probability for current word
        /// (if current grammeme key exists in category otherwise null)
        /// </summary>
        /// <param name="key">Grammeme key</param>
        public Gram this[string key]
        {
            get
            {
                var tag = _grams.FirstOrDefault(x => x.Key == key);
                if (tag == null)
                    return new Gram(key, 0);
                
                return tag;
            }
        }       
    }
}