using System.Collections.Generic;

namespace AmbDataset
{
    public class Token
    {
        public string Text { get; set; }
        public List<string> Grams { get; } = new List<string>();
        public int[] Tags { get; set; }
    }
}