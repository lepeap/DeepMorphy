namespace DeepMorphy.WordDict
{
    internal class Word
    {
        public Word(string text, int tagId, bool replaceOther=false, string lemma = null)
        {
            Text = text;
            TagId = tagId;
            Lemma = lemma;
            ReplaceOther = replaceOther;
        }
            
        public string Text { get; }
            
        public int TagId { get; }
        
        public bool ReplaceOther { get;  }
        public string Lemma { get; set; }

        public override string ToString()
        {
            return Lemma == null ? $"{Text}[{TagId}]" : $"{Text}[{TagId}] - {Lemma}";
        }
    }
}