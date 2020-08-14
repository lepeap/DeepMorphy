using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace DeepMorphy
{
    /// <summary>
    /// Комбинация граммем для данного слова
    /// --------------------
    /// Combination of grammemes for word
    /// </summary>
    public sealed class Tag
    {
        internal Tag(ReadOnlyDictionary<string, string> gramsDic, float power, string lemma=null, int? tagIndex = null)
        {
            GramsDic = gramsDic;
            Power = power;
            TagIndex = tagIndex;
            Lemma = lemma;
        }

        /// <summary>
        /// Словарь, ключ грамматической категории -> значение граммемы 
        /// --------------------
        /// Gram category key -> Gram value dictionary
        /// </summary>
        public ReadOnlyDictionary<string, string> GramsDic { get; }

        /// <summary>
        /// Перечисление граммем для данного слова
        /// --------------------
        /// Enumeration of grammemes keys for current word
        /// </summary>
        public IEnumerable<string> Grams => GramsDic.Values;
        
        /// <summary>
        /// Вероятность данной комбинации
        /// --------------------
        /// Probability of current combination
        /// </summary>
        public float Power { get; }
        
        /// <summary>
        /// Лемма для данной комбинации 
        /// --------------------
        /// Lemma for current combination
        /// </summary>
        public string Lemma { get; internal set; }

        /// <summary>
        /// Задана ли лемма в данном теге
        /// --------------------
        /// Is lemma set in current tag
        /// </summary>
        public bool HasLemma => Lemma != null;
        
        //TODO: Написать комменты
        public string Post => _getWithMultilangKey("post");

        public string Gender => _getWithMultilangKey("gndr");
        
        public string Number => _getWithMultilangKey("nmbr");
        
        public string Case => _getWithMultilangKey("case");
        
        public string Tens => _getWithMultilangKey("tens");

        public string Mood => _getWithMultilangKey("mood");
        
        public string Pers => _getWithMultilangKey("pers");

        /// <summary>
        /// Проверяет, есть ли граммемы в данном теге
        /// --------------------
        /// Checks for grammemes in current tag
        /// </summary>
        /// <param name="grams">
        /// Ключи граммем для проверки
        /// --------------------
        /// Grammeme keys to check
        /// </param>
        /// <returns>
        /// true, если все перечисленные граммемы присутсвубт в теге
        /// --------------------
        /// true if current tag contains all this grammemes else false
        /// </returns>
        public bool Has(params string[] grams)
        {
            return grams.All(gram => Grams.Contains(gram));
        }
        
        /// <summary>
        /// Возвращает значение грамматической категории
        /// --------------------
        /// Returns grammeme for grammatical category
        /// </summary>
        /// <param name="gramCatKey">
        /// Ключ грамматической категории
        /// --------------------
        /// Grammatical category key
        /// </param>
        public string this[string gramCatKey]
        {
            get
            {
                if (GramsDic.ContainsKey(gramCatKey))
                {
                    return GramsDic[gramCatKey];
                }
                
                return null;
            }
        }

        public override string ToString()
        {
            var tags = string.Join(",", Grams);
            if (Lemma == null)
            {
                return tags;
            }
            
            return $"Lemma: {Lemma} Tags: {tags}";
        }
        
        internal int? TagIndex { get; }


        private string _getWithMultilangKey(string enKey)
        {
            if (GramsDic.ContainsKey(enKey))
            {
                return this[enKey];
            }

            return this[GramInfo.EnRuDic[enKey]];
        }
    }
}