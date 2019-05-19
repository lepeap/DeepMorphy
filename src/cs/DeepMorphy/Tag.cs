using System.Linq;

namespace DeepMorphy
{
    public sealed class Tag
    {
        internal Tag(string[] grams, float power, string lemma=null, int? classIndex = null)
        {
            Grams = grams;
            Power = power;
            ClassIndex = classIndex;
            Lemma = lemma;
        }
        public string[] Grams { get; }
        public float Power { get; }
        
        public string Lemma { get; }
        
        internal int? ClassIndex { get; }

        public bool Has(string tag)
        {
            return Grams.Contains(tag);
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