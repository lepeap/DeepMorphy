namespace DeepMorphy
{
    /// <summary>
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
        /// Grammeme key
        /// </summary>
        public string Key { get; }
        
        /// <summary>
        /// Probability for current grammeme in grammatical category for current word
        /// </summary>
        public float Power { get; }
        
        public override string ToString()
        {
            return $"{Key} : {Power}";
        }
    }
}