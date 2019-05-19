using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace DeepMorphy
{
    public sealed class GramCategory
    {
        private Gram[] _grams;
        internal GramCategory(Gram[] grams)
        {
            _grams = grams;
        }

        public Gram BestGram => _grams[0];

        public IEnumerable<Gram> Grams => _grams;

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