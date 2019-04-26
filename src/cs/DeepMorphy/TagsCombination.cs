using System.Linq;

namespace DeepMorphy
{
    public sealed class TagsCombination
    {
        internal TagsCombination(string[] tags, float power)
        {
            Tags = tags;
            Power = power;
        }
        public string[] Tags { get; }
        public float Power { get; }

        public bool Has(string tag)
        {
            return Tags.Contains(tag);
        }

        public override string ToString()
        {
            return string.Join(",", Tags);
        }
    }
}