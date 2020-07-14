namespace DeepMorphy
{
    public class TagHelper
    {
        private readonly MorphAnalyzer _morph;
        internal TagHelper(MorphAnalyzer morph)
        {
            _morph = morph;
        }
        
        public Tag CreateForInfn(string word, string lemma=null)
        {
            return null;
        }
        
    }
}