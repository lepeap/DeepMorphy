namespace DeepMorphy.PreProc
{
    internal interface IPreProcessor
    {
        MorphInfo Parse(string word);
    }
}