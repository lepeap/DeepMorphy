namespace DeepMorphy
{
    public sealed class Gram
    {
        internal Gram(string key, float power)
        {
            Key = key;
            Power = power;
        }
        public string Key { get; }
        public float Power { get; }
        
        
        public override string ToString()
        {
            return $"{Key} : {Power}";
        }
    }
}