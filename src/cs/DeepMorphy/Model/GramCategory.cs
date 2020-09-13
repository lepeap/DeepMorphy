using System.Collections.Generic;
using System.Linq;

namespace DeepMorphy.Model
{
    /// <summary>
    /// Грамматическая категория, содержит вероятность каждой граммемы в категории для текущего слова
    /// --------------------
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
        /// Самая вероятная граммема
        /// --------------------
        /// Best grammeme in category for current word
        /// </summary>
        public Gram BestGram => _grams[0];
        
        /// <summary>
        /// Ключ самой вероятной граммемы 
        /// --------------------
        /// Best grammeme key in category for current word
        /// </summary>
        public string BestGramKey => _grams[0].Key;
        
        /// <summary>
        /// Все граммемы с вероятностями в данной категории 
        /// --------------------
        /// All grammemes with probabilities for current word
        /// </summary>
        public IEnumerable<Gram> Grams => _grams;
        
        /// <summary>
        /// Возвращает граммему и ее вероятность для данного слова
        /// (если граммема отсутствует в категории, то null)
        /// --------------------
        /// Returns grammeme and probability for current word
        /// (if current grammeme key exists in category otherwise null)
        /// </summary>
        /// <param name="key">
        /// Ключ граммемы
        /// --------------------
        /// Grammeme key
        /// </param>
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