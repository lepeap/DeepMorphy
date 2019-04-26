using DeepMorphy.WordDict;

namespace DeepMorphy.PreProc
{
    class DictProc : IPreProcessor
    {
        private readonly Dict _dict;
        public DictProc(Dict dict)
        {
            _dict = dict;
        }

        public Token Parse(string word)
        {
            return _dict.Parse(word);
        }
    }
}