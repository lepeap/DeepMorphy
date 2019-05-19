using System.Linq;

namespace DeepMorphy
{
    /// <summary>
    /// Combination of grammemes for token
    /// </summary>
    public sealed class Tag
    {
        internal Tag(string[] grams, float power, string lemma=null, int? classIndex = null)
        {
            Grams = grams;
            Power = power;
            ClassIndex = classIndex;
            Lemma = lemma;
        }

        internal int? ClassIndex { get; }
        
        /// <summary>
        /// Array of grammemes keys for current word
        /// </summary>
        public string[] Grams { get; }
        
        /// <summary>
        /// Probability for current combination
        /// </summary>
        public float Power { get; }
        
        /// <summary>
        /// Lemma for current combination
        /// </summary>
        public string Lemma { get; }
        
        /// <summary>
        /// Checks for grammeme in current tag
        /// </summary>
        /// <param name="gram">Grammeme</param>
        /// <returns>true if current tag contains this grammeme else false</returns>
        public bool Has(string gram)
        {
            return Grams.Contains(gram);
        }
        
        public override string ToString()
        {
            var tags = string.Join(",", Grams);
            if (Lemma == null)
                return tags;
            else
                return $"Lemma: {Lemma} Tags: {tags}";
        }

    }
}