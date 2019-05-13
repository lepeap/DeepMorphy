using System.Linq;

namespace DeepMorphy
{
    public sealed class TagsCombination
    {
        internal TagsCombination(string[] tags, float power, string lemma=null, int? classIndex = null)
        {
            Tags = tags;
            Power = power;
            ClassIndex = classIndex;
            Lemma = lemma;
        }
        public string[] Tags { get; }
        public float Power { get; }
        
        public string Lemma { get; internal set; }
        
        internal int? ClassIndex { get; }

        public bool Has(string tag)
        {
            return Tags.Contains(tag);
        }

        public override string ToString()
        {
            var tags = string.Join(",", Tags);
            if (Lemma == null)
                return tags;
            else
                return $"Lemma: {Lemma} Tags: {tags}";
        }

    }
}