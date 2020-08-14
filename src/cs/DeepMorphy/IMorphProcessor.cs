using System.Collections.Generic;

namespace DeepMorphy
{
    internal interface IMorphProcessor
    {
        IEnumerable<MorphInfo> Parse(IEnumerable<string> words);
        IEnumerable<string> Inflect(IEnumerable<(string word, Tag wordTag, Tag resultTag)> tasks);
        IEnumerable<(Tag tag, string text)> Lexeme(string word, Tag tag);
    }
}