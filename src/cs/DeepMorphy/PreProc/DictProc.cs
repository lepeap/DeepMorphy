using DeepMorphy.WordDict;

namespace DeepMorphy.PreProc
{
    internal class DictProc : IPreProcessor
    {
        private readonly Dict _dict;
        public DictProc(Dict dict)
        {
            _dict = dict;
        }

        public MorphInfo Parse(string word)
        {
            return _dict.Parse(word);
        }
    }
}