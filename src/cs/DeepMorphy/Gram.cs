namespace DeepMorphy
{
    /// <summary>
    /// Значение граммемы для слова
    /// --------------------
    /// Grammeme value for word
    /// </summary>
    public sealed class Gram
    {
        internal Gram(string key, float power)
        {
            Key = key;
            Power = power;
        }
        
        /// <summary>
        /// Ключ граммемы
        /// --------------------
        /// Grammeme key
        /// </summary>
        public string Key { get; }
        
        /// <summary>
        /// Вероятность граммемы
        /// --------------------
        /// Probability for current grammeme in grammatical category for current word
        /// </summary>
        public float Power { get; }
        
        public override string ToString()
        {
            return $"{Key} : {Power}";
        }
    }
}