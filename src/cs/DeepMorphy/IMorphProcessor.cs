using System.Collections.Generic;

namespace DeepMorphy
{
    internal interface IMorphProcessor
    {
        string Key { get; }
        bool IgnoreNetworkResult { get; }
        IEnumerable<(int tagId, string lemma)> Parse(string word);
        string Lemmatize(string word, int tagId);
        string Inflect(string word, int wordTag, int resultTag);
        IEnumerable<(int tagId, string text)> Lexeme(string word, int tag);
    }
}