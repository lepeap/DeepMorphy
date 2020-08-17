using System.Collections.Generic;

namespace DeepMorphy
{
    internal interface IMorphProcessor
    {
        IEnumerable<(int tagId, string lemma)> Parse(string word);
        string Inflect(string word, int wordTag, int resultTag);
        IEnumerable<(int tag, string text)> Lexeme(string word, int tag);
    }
}